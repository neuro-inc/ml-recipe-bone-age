import argparse
import time
import logging
import json
import shutil
import asyncio
import zipfile
import requests
import io
from urllib.parse import urlparse, quote
from pathlib import Path
from typing import Optional, List

import pandas as pd

logging.basicConfig(level=logging.INFO)
ENCODINGS = {
    'gender': {
        'G': False,
        'B': True,
    }
}


def main(args: argparse.Namespace) -> None:
    cmd = ["label-studio"] + args.label_studio_args
    logging.info(f"Launching label studio with cmd: '{' '.join(cmd)}' ")
    loop = asyncio.get_event_loop()
    return_code = loop.run_until_complete(
        _run_label_studio(
            cmd,
            ls_project_root=args.ls_project_root,
            project_root=args.project_root,
        )
    )
    return return_code


async def _run_label_studio(
    cmd: List[str], ls_project_root: Path, project_root: Path
) -> None:
    ls_proc = await asyncio.create_subprocess_exec(*cmd)
    while True:
        if _all_tasks_finished(ls_project_root, project_root):
            logging.warning(
                "All tasks are finished! Uploading labels and terminating Label-studio."
            )
            _save_labeling_results(project_root, _find_ls_port(ls_project_root, cmd))
            if ls_proc.returncode is None:
                ls_proc.terminate()
            await ls_proc.wait()
            _migrate_uploaded_completions(ls_project_root, project_root)
            break
        else:
            time.sleep(1)


def _all_tasks_finished(ls_project_root: Path, project_root: Path) -> bool:
    completions_dir = ls_project_root / "completions"
    imgs_dir = project_root / "tmp_data" / "images"
    uploaded_imgs_dir = ls_project_root / "upload"
    completions = 0  # finished labels
    imgs = 0  # images in dataset
    uploaded_imgs = 0  # uploaded images into label studio
    if completions_dir.exists() and completions_dir.is_dir():
        completions = len(list(completions_dir.glob("*.json")))
    if imgs_dir.exists() and imgs_dir.is_dir():
        imgs = len(list(imgs_dir.iterdir()))
    if uploaded_imgs_dir.exists() and uploaded_imgs_dir.is_dir():
        uploaded_imgs = len(list(uploaded_imgs_dir.iterdir()))
    if completions >= imgs + uploaded_imgs:
        return True  # all tasks, including the uploaded ones are annotated, terminating
    else:
        return False


def _save_labeling_results(project_root: Path, label_studio_port: str) -> None:
    addr = f"http://localhost:{label_studio_port}/api/project/export?format=CSV"
    for retry in range(5, 0, -1):
        try:
            response = requests.get(addr)
            results_archive = zipfile.ZipFile(io.BytesIO(response.content))
            results = results_archive.read("result.csv")
            target_csv_file = project_root / "data" / "annotations.csv"
            _save_results(target_csv_file, io.BytesIO(results))
            break
        except requests.RequestException as e:
            logging.warning(f"Retry {retry}: cannot retrieve results from {addr}: {e}")
            time.sleep(2)


def _save_results(target_csv_file: Path, source: io.BytesIO):
    """
    Example of result out of Label-Studio:
       image,                                                 id,gender,boneage
    0  /data/2012.png?d=%2Flabel-studio%2Fmy_project%2Fimages,1,G,"[{""text"": [""4""]}]"
    1  /data/2015.png?d=%2Flabel-studio%2Fmy_project%2Fimages,2,G,"[{""text"": [""41""]}]"
    2  /data/2104.png?d=%2Flabel-studio%2Fmy_project%2Fimages,3,B,"[{""text"": [""41""]}]"

    Example of target results:
           id  boneage   male  boneage_years
    53   1437      136  False      11.333333
    56   1440       42   True       3.500000
    84   1472       60   True       5.000000
    125  1516      132   True      11.000000
    146  1541       24   True       2.000000
    """
    target_csv = pd.read_csv(target_csv_file, index_col=[0])
    source_csv = pd.read_csv(source)
    source_transformed = source_csv.apply(_convert_row, axis=1)
    source_transformed = source_transformed.astype(target_csv.dtypes)
    source_transformed.index = source_transformed.pop("id")
    merged = pd.concat([target_csv, source_transformed])
    merged.to_csv(target_csv_file, index=False)


def _convert_row(r: pd.Series) -> pd.Series:
    gender = r["gender"]
    result = pd.Series()
    result["male"] = ENCODINGS['gender'][gender]
    boneage = json.loads(r["boneage"])  # [{'text': ['12']}]
    boneage = int(boneage[0]["text"][0])
    result["boneage"] = boneage
    result["boneage_years"] = boneage / 12
    result["id"] = Path(urlparse(r["image"]).path).with_suffix("").name

    return result


def _find_ls_port(ls_project_root: Path, launch_cmd: List[str]) -> str:
    if "--port" in launch_cmd:
        port = launch_cmd[launch_cmd.index("--port") + 1]
    else:
        try:
            ls_config = json.loads((ls_project_root / "config.json").read_text())
            port = ls_config.get("port")
        except Exception:
            port = None
        if not port:
            # Default for Label-studio, if nothing else is provided.
            port = 8080  # type: ignore
    return str(port)


def _migrate_uploaded_completions(ls_project_root: Path, project_root: Path) -> None:
    "migrate completions for uploaded tasks via web GUI"
    upload_dir = ls_project_root / "upload"
    if not upload_dir.exists() or not upload_dir.is_dir():
        logging.info(f"Upload dir was not found under {str(upload_dir.resolve())}.")
        return
    complitions_dir = ls_project_root / "completions"
    for completion_p in complitions_dir.iterdir():
        completion = json.loads(completion_p.read_text())
        if "task_path" in completion.keys():
            # it is not an uploaded task
            continue
        task_url = urlparse(completion["data"]["image"])
        f_name = Path(task_url.path).name
        data_root_abs = (project_root / "data" / "images").resolve()

        completion["task_path"] = str(data_root_abs / f_name)
        img_url = f"/data/{f_name}?d={quote(str(data_root_abs), safe='')}"
        completion["data"]["image"] = img_url
        shutil.move(upload_dir / f_name, data_root_abs / f_name)  # type: ignore

        with open(completion_p, "w") as completion_fd:
            json.dump(completion, completion_fd, indent=2)
        logging.info(f"Migrated uploaded completion {str(completion_p), }")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "label_studio_args",
        nargs="*",
        help="Args passed to launch label-studio",
    )
    parser.add_argument(
        "-r",
        "--project_root",
        type=Path,
        help="Project root path (loaded from GH)",
    )
    parser.add_argument(
        "-l",
        "--ls_project_root",
        type=Path,
        help="Label studio project root path",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main(get_args())
    # _save_labeling_results(Path("."), "8080")
