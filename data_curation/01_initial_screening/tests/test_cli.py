from medpmc_initial_screening.cli import build_parser


def test_prepare_bulk_command_parses():
    args = build_parser().parse_args(
        [
            "prepare-bulk",
            "--archive-url",
            "https://example.org/archive.tar.gz",
            "--filelist-url",
            "https://example.org/filelist.csv",
            "--output-dir",
            "outputs/example",
        ]
    )
    assert args.command == "prepare-bulk"
    assert args.shard_size == 10_000


def test_prepare_pmcids_command_parses():
    args = build_parser().parse_args(
        [
            "prepare-pmcids",
            "--pmcid-file",
            "examples/pmcids.txt",
            "--output-dir",
            "outputs/example",
        ]
    )
    assert args.command == "prepare-pmcids"
    assert args.workers == 8


def test_screen_reference_mode_defaults_to_model_compatible():
    args = build_parser().parse_args(
        [
            "screen",
            "--input-dir",
            "work/intermediate/figure_text",
            "--output-dir",
            "work/results",
        ]
    )
    assert args.reference_mode == "model-compatible"


def test_screen_reference_mode_can_use_clean_text():
    args = build_parser().parse_args(
        [
            "screen",
            "--input-dir",
            "work/intermediate/figure_text",
            "--output-dir",
            "work/results",
            "--reference-mode",
            "clean",
        ]
    )
    assert args.reference_mode == "clean"
