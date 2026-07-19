from medpmc_multi_panel_figure_separation.cli import build_parser


def test_cli_defaults():
    args = build_parser().parse_args(
        ["separate", "--classified-dir", "classified", "--output-dir", "out"]
    )
    assert args.conf == 0.5
    assert args.batch_size == 1
    assert args.device == "auto"
