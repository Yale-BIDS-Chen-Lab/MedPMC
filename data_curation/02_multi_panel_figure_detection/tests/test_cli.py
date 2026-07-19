from medpmc_multi_panel_figure_detection.cli import build_parser


def test_prepare_images_parser():
    args = build_parser().parse_args(
        ["prepare-images", "--input-dir", "retained", "--output-dir", "out"]
    )
    assert args.command == "prepare-images"
    assert args.workers == 16


def test_detect_parser_defaults():
    args = build_parser().parse_args(
        ["detect", "--manifest-dir", "manifest", "--output-dir", "results"]
    )
    assert args.command == "detect"
    assert args.image_size is None
    assert args.positive_label == 1
