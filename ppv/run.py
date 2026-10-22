"""Usage: python -m ppv.run --config ppv/config.local.json"""

import argparse
import sys

from ppv.config import read_config
from ppv.pipeline import run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Standalone base/PPV paired-video correction verification")
    parser.add_argument("--config", required=True, help="JSON configuration; relative paths resolve against this file")
    args = parser.parse_args(argv)
    try:
        config = read_config(args.config)
    except (OSError, ValueError, TypeError) as error:
        # Do not echo malformed configuration values that may contain credentials.
        print(f"Could not load configuration ({type(error).__name__}); check JSON fields and file path.", file=sys.stderr)
        return 1
    directory, success = run(config)
    print(f"{'Success' if success else 'Failed'}: {directory / 'result.json'}", flush=True)
    if not success:
        print(f"See {directory / 'run.log'} for the failure reason.", file=sys.stderr)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
