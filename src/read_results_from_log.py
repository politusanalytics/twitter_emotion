import sys
import re
import numpy as np
import pandas as pd

input_log_filename = sys.argv[1]
max_or_mean = "max"
if max_or_mean == "max_special":
    metric_names = ["f1_macro", "f1_micro", "mcc", "precision_macro", "precision_micro", "recall_macro", "recall_micro"]
    labels = ["category1", "category2", "category3"]

if __name__ == "__main__":
    with open(input_log_filename, "r") as log_file:
        all_conf_results = {}
        all_metrics = []

        curr_conf_results = []
        curr_seed_results = {}
        results_lines = False
        conf_name = ""
        for line in log_file:
            if "======" in line: # start of current configuration's results
                # write previous configuration's results
                if len(curr_conf_results) > 0:
                    all_conf_results[conf_name] = curr_conf_results

                conf_name = re.search("====== ([^=]+) ======", line).group(1)
                curr_conf_results = []

            elif "***** TEST RESULTS *****" in line:
                results_lines = True

            elif "TEST SCORE: " in line:
                # # check if metrics match
                # if len(all_conf_results) > 0 or len(curr_conf_results) > 0:
                #     assert(sorted(list(curr_seed_results.keys())) == all_metrics)
                # else:
                #     all_metrics = sorted(list(curr_seed_results.keys()))

                all_metrics += list(curr_seed_results.keys())
                curr_conf_results.append(curr_seed_results)
                curr_seed_results = {}
                results_lines = False

            elif results_lines:
                match = re.search(r"(\w+) = (-?\d{1,2}\.\d+)$", line)
                result_name = match.group(1)
                curr_result = float(match.group(2)) * 100
                curr_seed_results[result_name] = curr_result

    if len(curr_conf_results) > 0:
        all_conf_results[conf_name] = curr_conf_results

    all_metrics = sorted(list(set(all_metrics)))
    results_df = pd.DataFrame(columns=all_metrics)
    for curr_conf_results in all_conf_results.values():
        metrics_results = {}
        if max_or_mean == "mean":
            for metric in all_metrics:
                curr_metric_results = np.array([seed_results[metric] for seed_results in curr_conf_results if seed_results.get(metric, "") != ""])
                if len(curr_metric_results) > 0:
                    curr_mean = round(np.mean(curr_metric_results), 4)
                    curr_std = round(np.std(curr_metric_results), 4)
                    metrics_results[metric] = str(curr_mean) + " +- " + str(curr_std)
                else:
                    metrics_results[metric] = "---"

            results_df = results_df.append(metrics_results, ignore_index=True)

        elif max_or_mean == "max":
            # NOTE: f1_macro might not exist
            max_seed_idx = np.argmax([seed_results["f1_macro"] for seed_results in curr_conf_results])
            max_seed_results = curr_conf_results[max_seed_idx]
            for metric in all_metrics:
                res = max_seed_results.get(metric, -1.0)
                metrics_results[metric] = "{:.4f}".format(res) if res != -1.0 else "---"

            results_df = results_df.append(metrics_results, ignore_index=True)

        elif max_or_mean == "max_special":
            assert(len(all_conf_results) == 1)
            results_df = pd.DataFrame(columns=metric_names)

            max_seed_idx = np.argmax([seed_results["f1_macro"] for seed_results in curr_conf_results])
            max_seed_results = curr_conf_results[max_seed_idx]

            for lab in labels:
                for metric_name in metric_names:
                    res = max_seed_results.get(f"{lab}_{metric_name}", -1.0)
                    metrics_results[metric_name] = "{:.4f}".format(res) if res != -1.0 else "---"
                results_df = results_df.append(metrics_results, ignore_index=True)

            # add mean of labels
            for metric_name in metric_names:
                res = max_seed_results.get(metric_name, -1.0)
                metrics_results[metric_name] = "{:.4f}".format(res) if res != -1.0 else "---"
            results_df = results_df.append(metrics_results, ignore_index=True)

        else:
            raise(f"Invalid value {max_or_mean} for max_or_mean!")

    if max_or_mean == "max_special":
        results_df.index = labels + ["all"]
    else:
        results_df.index = list(all_conf_results.keys())

    results_df.to_html(input_log_filename + ".html")
