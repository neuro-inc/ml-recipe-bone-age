import argparse
import shutil
from pathlib import Path
import logging

import pandas as pd

logger = logging.getLogger()


def extend_dataset(args: argparse.Namespace) -> None:
    cur_dataset = Path(args.cur_dataset)
    full_dataset = Path(args.full_dataset)
    nmber_of_imgs = int(args.nmber_of_imgs)

    assert full_dataset.is_dir(), (full_dataset, "not exists")
    assert nmber_of_imgs > 0, (nmber_of_imgs, "not positive number")

    full_images_dir = full_dataset / "images"
    full_annotations_csv = full_dataset / "annotations.csv"
    assert full_images_dir.is_dir(), (full_images_dir, "not exists")
    assert full_annotations_csv.is_file(), (full_annotations_csv, "not exists")

    cur_images_dir = cur_dataset / "images"
    cur_annotations_csv = cur_dataset / "annotations.csv"

    full_df = pd.read_csv(full_annotations_csv, index_col=0)
    max_nmber_of_imgs = len(full_df)
    if nmber_of_imgs >= max_nmber_of_imgs:
        logger.warning(f"Changing number of images {nmber_of_imgs} -> {max_nmber_of_imgs}")
        nmber_of_imgs = max_nmber_of_imgs
    logger.info(f"Loaded {len(full_df)} items from full dataset {full_annotations_csv}")
    take_df, rest_df = full_df[:nmber_of_imgs], full_df[nmber_of_imgs:]
    logger.info(f"Split full({len(full_df)}) -> take({len(take_df)}) + rest({len(rest_df)})")

    if cur_annotations_csv.exists():
        cur_df = pd.read_csv(cur_annotations_csv, index_col=0)
        logger.info(f"Loaded {len(cur_df)} items from current dataset {cur_annotations_csv}")
        result_df = pd.concat([cur_df, take_df], sort=False)#, axis='columns')
        if not args.skip_annotation_update:
            logger.info(f"Will merge current({len(cur_df)}) + take({len(take_df)}) -> result({len(result_df)})")
    else:
        logger.info(f"Current dataset not found: {cur_annotations_csv}")
        result_df = take_df
        logger.info(f"Will use take({len(take_df)}) -> result({len(result_df)})")

    img_ids_to_move = take_df['id'].to_list()
    img_names_to_move = [f"{id}.png" for id in img_ids_to_move]
    img_paths_to_move = [full_images_dir / name for name in img_names_to_move]
    logger.info(f"Expected images to move: {len(img_paths_to_move)}")
    img_paths_to_move = [path for path in img_paths_to_move if path.exists()]
    logger.info(f"Found images to move: {len(img_paths_to_move)}")

    logger.info(f"Moving {len(img_paths_to_move)} images from {full_images_dir} to {cur_images_dir}")
    cur_images_dir.mkdir(exist_ok=True, parents=True)
    for img_path in img_paths_to_move:
        shutil.copy(str(img_path), str(cur_images_dir/img_path.name))

    if not args.skip_annotation_update:
        logger.info(f"Writing result({len(result_df)}) to current dataset {cur_annotations_csv}")
    cur_annotations_csv.write_text(result_df.to_csv())

    logger.info(f"Writing rest({len(rest_df)}) to full dataset {full_annotations_csv}")
    full_annotations_csv.write_text(rest_df.to_csv())

    logger.info(f"[+] Success!")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--cur_dataset",
        required=True,
        type=Path,
        help="Path to current data",
    )
    parser.add_argument(
        "-f",
        "--full_dataset",
        required=True,
        type=Path,
        help="Path to full dataset, from where new images will be taken",
    )
    parser.add_argument(
        "-n",
        "--nmber_of_imgs",
        default=1,
        type=int,
        help="How many new images to add from dataset into current data",
    )
    parser.add_argument(
        "--skip_annotation_update",
        default=True,
        action='store_false',
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    extend_dataset(get_parser().parse_args())
