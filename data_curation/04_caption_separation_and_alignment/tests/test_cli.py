from medpmc_caption_separation_and_alignment.cli import build_parser


def _required(command: str) -> list[str]:
    return [
        command,
        "--parent-dir",
        "parents",
        "--panel-dir",
        "panels",
        "--parent-image-root",
        "stage2",
        "--panel-image-root",
        "stage3",
        "--output-dir",
        "out",
    ]


def test_cli_defaults_and_run_command():
    args = build_parser().parse_args(_required("run"))
    assert args.command == "run"
    assert args.action == "run"
    assert args.model == "Yale-BIDS-Chen/medpmc-caption-separation-internvl-2.5-4b-mpo"
    assert args.session_len == 32768
    assert args.max_new_tokens == 1024
    assert args.retry_truncated is False
    assert args.retry_max_new_tokens == 2048
    assert args.tp == 1
    assert args.trust_remote_code is None


def test_align_remains_a_compatibility_alias():
    args = build_parser().parse_args(_required("align"))
    assert args.command == "align"
    assert args.action == "run"


def test_retry_and_merge_commands():
    retry = build_parser().parse_args([
        "retry",
        "--source-run-dir", "first",
        *_required("run")[1:],
    ])
    assert retry.action == "retry"
    assert retry.max_new_tokens == 2048
    assert retry.status is None

    merge = build_parser().parse_args([
        "merge",
        "--base-run-dir", "first",
        "--retry-run-dir", "retry2048",
        "--output-dir", "final",
    ])
    assert merge.action == "merge"
    assert merge.retry_run_dir == ["retry2048"]
