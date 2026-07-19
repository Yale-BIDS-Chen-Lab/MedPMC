from medpmc_medical_figure_classification.cli import build_parser


def test_cli_defaults():
    parser = build_parser()
    args = parser.parse_args(["run", "--output-dir", "out", "--singlepanel-dir", "in", "--singlepanel-image-root", "root"])
    assert args.batch_size == 256
    assert args.threshold == 0.5
    assert args.medical_label == 1
    assert args.amp is False
