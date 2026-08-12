import argparse
import analyser

def create_parser():
    metrics = analyser.get_metrics()
    parser = argparse.ArgumentParser(prog="VideoMetricsTool", description="Calculates a set of NR quality metrics on a given video")
    parser.add_argument('path', help="Path to the video to be analyzed")
    parser.add_argument('-o', '--output', default="output.csv", help="Path to the desired output csv-file. If not provided, \"output.csv\" is created within the same directory as the main script.")
    parser.add_argument('-i', '--include', default=metrics, help="The NR features you would like to calculate. If not provided, all of them are calculated.", nargs="+", choices=metrics)
    parser.add_argument('-e', '--exclude', default=[], help="Any NR metrics you would like to exclude from the default set (e.g. to speed up calculations). If not provided, all metrics are calculated.", choices=metrics, nargs="+")
    #parser.add_argument('-a', '--aggregate', action='store_true', help="If flagged, metrics are returned as an aggregation over the duration of the video. Otherwise, a per frame output is provided.")
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    print(args)
    metrics = list(set(args.include) - set(args.exclude))
    analyser.analyse_video(args.path, metrics, args.output)

if __name__ == '__main__':
    main()
