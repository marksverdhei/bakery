def pytest_addoption(parser):
    parser.addoption(
        "--benchmark-compare",
        default=None,
        help="Git ref (branch/tag/sha) to compare against using a worktree",
    )
    parser.addoption(
        "--benchmark-save",
        default=None,
        help="Save benchmark results to this JSON file",
    )
