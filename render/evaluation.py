import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

import time
import csv
import os

import pathlib
from pathlib import Path

from scene.ensemble import Ensemble
from render.__main__ import easeInOutCubic

class Evaluation:

    profile1 = {
        "outputfile": "evaluation_al_octreedepth.csv",
        "directory": pathlib.Path(__file__).parents[3] / "data/skull/al/4m",

        "octree_node_size": 400,
        "normalize_data": False,
        "autograd": True,
        "sort_emd": False,
        "uniform_reference": False,
        "reference_n": 0,

        "blur" : 0.0005,
        "scaling" : 0.7,
        "truncate" : 7,
        "reach": 1,

        "min_sample_size": 300000,
        "max_sample_size": 800000,
        "num_samplings": 6,
        
        "varying_parameter": "octree_node_size",
        #"varying_values": [200, 400, 600, 800, 1000, 2000, 5000, 10000],
        "varying_values": [200],

        #"varying_parameter": "truncate",
        #"varying_values": [0.1, 0.5, 1, 3, 5, 7, 10],

        #"varying_parameter": "reach",
        #"varying_values": [0.1, 0.5, 1, 5, 10],
    }

    profile2 = {
        "outputfile": "evaluation_glacier.csv",
        "directory": pathlib.Path(__file__).parents[3] / "data/glacier/segmentation_3/raw",

        "octree_node_size": 400,
        "normalize_data": False,
        "autograd": True,
        "sort_emd": False,
        "uniform_reference": False,
        "reference_n": 0,

        "blur" : 0.001,
        "scaling" : 0.7,
        "truncate" : 1,
        "reach": 2,

        "min_sample_size": 100000,
        "max_sample_size": 12000000,
        "num_samplings": 20,
        
        "varying_parameter": "octree_node_size",
        "varying_values": [2000],
        #"varying_values": [2000],

        #"varying_parameter": "truncate",
        #"varying_values": [0.1, 0.5, 1, 3, 5, 7, 10],

        #"varying_parameter": "reach",
        #"varying_values": [0.1, 0.5, 1, 5, 10],
    }

    profile3 = {
        "outputfile": "evaluation_al_naive.csv",
        "directory": pathlib.Path(__file__).parents[3] / "data/skull/al/4m",

        "octree_node_size": 200,
        "normalize_data": False,
        "autograd": True,
        "sort_emd": False,
        "uniform_reference": False,
        "reference_n": 0,

        "blur" : 0.0005,
        "scaling" : 0.7,
        "truncate" : 7,
        "reach": 1,

        "min_sample_size": 100000,
        "max_sample_size": 1500000,
        "num_samplings": 20,
        
        "varying_parameter": "truncate",
        #"varying_values": [200, 400, 600, 800, 1000, 2000, 5000, 10000],
        "varying_values": [7],

        #"varying_parameter": "truncate",
        #"varying_values": [0.1, 0.5, 1, 3, 5, 7, 10],

        #"varying_parameter": "reach",
        #"varying_values": [0.1, 0.5, 1, 5, 10],
    }

    #def __init__(self):
        #self.profile = self.profile1 
        # 
        #self.create_filelist()
        #infile_pth = str((Path.home() / "Dev" / self.profile["outputfile"]).resolve())
        #csv_to_latex_document(infile_pth, infile_pth.replace(".csv", ".tex"), caption="CSV Table", label="tab:csv_table")
        #csv_to_heatmap(infile_pth, infile_pth.replace(".csv", ".png"))
        #csv_to_lineplot(infile_pth, infile_pth.replace(".csv", ".png"))
        #csv_to_lineplot_parameter(infile_pth, infile_pth.replace(".csv", ".png"))

    def run(self):
        self.eval_only()
        infile_pth = str((Path.home() / "Dev" / self.profile["outputfile"]).resolve())
        #self.image_only(infile_pth)

    def eval_only(self):
        self.profile = self.profile3
        self.create_filelist()

    def image_only(self, path):
        #csv_to_heatmap(path, path.replace(".csv", ".png"))
        #csv_to_lineplot(path, path.replace(".csv", ".png"))
        csv_to_lineplot_parameter(path, path.replace(".csv", ".png"))

    def create_filelist(self):
        #folders = list_subfolders(self.profile["directory"])
        #for folder in folders:
        #files = list_filepaths(folder)
        files = list_filepaths(self.profile["directory"])
        files = [file for file in files if file.endswith(('.e57'))]
        print(f" files: {files}")

        #ensemble_subsamples = range(2, len(files))
        ensemble_subsamples = [2]

        for i in ensemble_subsamples:
            current_files = files[:i]
            filelist = {"files": current_files}

            print(f"Run with {i} files.")

            conf = {
                "octree_node_size": self.profile["octree_node_size"],
                "normalize_data": self.profile["normalize_data"],
                "autograd": self.profile["autograd"],
                "sort_emd": self.profile["sort_emd"],  
                "uniform_reference": False,
                "reference_n": 0,
                #"accumulate_distance": self.accumulate_distance
                "blur" : self.profile["blur"],
                "scaling" : self.profile["scaling"],
                "truncate" : self.profile["truncate"],
                "reach": self.profile["reach"]
            }

            #for value in reversed(self.profile["varying_values"]):
            for value in self.profile["varying_values"]:

                print(f"Run with {self.profile['varying_parameter']} = {value}.")

                conf[self.profile["varying_parameter"]] = value
                self.run_shape_shift(filelist, conf)
                #self.run_shape_shift_naive(filelist, conf)

    
    def run_shape_shift(self, filelist, conf):
        number_of_files = len(filelist['files']) + (1 if self.profile['uniform_reference'] else 0)

        #sample_sizes = np.geomspace(self.profile['min_sample_size'], self.profile['max_sample_size'], 15, endpoint=True, dtype=int)
        sample_sizes = np.flip(np.linspace(self.profile['min_sample_size'], self.profile['max_sample_size'], self.profile['num_samplings'], endpoint=True, dtype=int))

        for n_points in sample_sizes: 

            print(f"Run with {n_points} points.")

            ens = Ensemble(filelist, conf)
            ens.idx_lut = np.arange(number_of_files)

            #
            start_time_pre_processing = time.time()
            #

            ens.build_evaluation(n_points)

            #
            end_time = time.time()
            delta_pre_processing = end_time - start_time_pre_processing
            start_time_ot = time.time()
            #

            ens.ot_reference(conf)

            #
            processing_times = ens.processing_times
            print(f"min, max, mean time: {np.round(np.min(processing_times), 4)}, {np.round(np.max(processing_times), 4)}, {np.round(np.mean(processing_times), 4)}")
            end_time = time.time()
            delta_ot = end_time - start_time_ot
            print(f"Elapsed time Pre-Processing, and OT: {delta_pre_processing:.4f}, {delta_ot:.4f}")
            delta_total = delta_ot + delta_pre_processing
            print(f"Elapsed time Pre-Processing and Sinkhorn: {delta_total:.4f} seconds")
            #

            outfile_pth = Path.home() / "Dev" / self.profile["outputfile"]
            file_exists = outfile_pth.is_file()

            with open(outfile_pth, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    header = ["Members", "Size", "Param", "Depth", "Pre", "OT", "Min", "Max", "Mean"]
                    writer.writerow(header)
                row = [number_of_files, 
                       int(np.mean(ens.get_num_points())), 
                       conf[self.profile['varying_parameter']], 
                       np.mean(ens.get_octree_depth()), 
                       np.round(delta_pre_processing, 3), 
                       np.round(delta_ot, 3), 
                       np.round(np.min(processing_times), 3), 
                       np.round(np.max(processing_times), 3), 
                       np.round(np.mean(processing_times), 3)]
                writer.writerow(row)

    def run_shape_shift_naive(self, filelist, conf):
        number_of_files = len(filelist['files']) + (1 if self.profile['uniform_reference'] else 0)

        #sample_sizes = np.geomspace(self.profile['min_sample_size'], self.profile['max_sample_size'], 15, endpoint=True, dtype=int)
        sample_sizes = np.flip(np.linspace(self.profile['min_sample_size'], self.profile['max_sample_size'], self.profile['num_samplings'], endpoint=True, dtype=int))

        for n_points in sample_sizes: 

            print(f"Run with {n_points} points.")

            ens = Ensemble(filelist, conf)
            ens.idx_lut = np.arange(number_of_files)

            #
            start_time_pre_processing = time.time()
            #

            ens.build_evaluation(n_points)

            #
            end_time = time.time()
            delta_pre_processing = end_time - start_time_pre_processing
            start_time_ot = time.time()
            #

            ens.ot_reference(conf)

            #
            oct_processing_times = ens.processing_times
            print(f"min, max, mean time: {np.round(np.min(oct_processing_times), 4)}, {np.round(np.max(oct_processing_times), 4)}, {np.round(np.mean(oct_processing_times), 4)}")
            end_time = time.time()
            oct_delta_ot = end_time - start_time_ot
            print(f"Elapsed time Pre-Processing, and OT: {delta_pre_processing:.4f}, {oct_delta_ot:.4f}")
            oct_delta_total = oct_delta_ot + delta_pre_processing
            print(f"Elapsed time Pre-Processing and Sinkhorn: {oct_delta_total:.4f} seconds")
            #

            #
            start_time_ot = time.time()


            ens.naive_ot_for_evaluation(conf)

            #
            naive_processing_times = ens.processing_times
            print(f"min, max, mean time: {np.round(np.min(naive_processing_times), 4)}, {np.round(np.max(naive_processing_times), 4)}, {np.round(np.mean(naive_processing_times), 4)}")
            end_time = time.time()
            naive_delta_ot = end_time - start_time_ot
            print(f"Elapsed time OT: {naive_delta_ot:.4f}")
            #

            outfile_pth = Path.home() / "Dev" / self.profile["outputfile"]
            file_exists = outfile_pth.is_file()

            with open(outfile_pth, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    header = ["Members", "Size", "Param", "Depth", "Oct", "Naive", "Truncate"]
                    writer.writerow(header)
                row = [number_of_files, 
                       int(np.mean(ens.get_num_points())), 
                       conf['octree_node_size'], 
                       np.mean(ens.get_octree_depth()), 
                       np.round(oct_delta_ot, 3), 
                       np.round(naive_delta_ot, 3),
                       conf[self.profile['varying_parameter']]]
                writer.writerow(row)
        
def list_filepaths(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))]

def list_subfolders(directory):
    return [os.path.join(directory, name) for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]

def csv_to_latex_document(csv_path, output_tex_path, caption="CSV Table", label="tab:csv_table"):

    df = pd.read_csv(csv_path)

    # Escape LaTeX special characters in column headers
    df.columns = df.columns.map(lambda x: str(x).replace('_', r'\_'))

    # Convert to LaTeX table string (booktabs for prettier lines, index=False to skip row index)
    table_latex = df.to_latex(index=False, escape=True, column_format='|'+ 'r|'*len(df.columns), header=True)

    # Wrap in full LaTeX document
    document = rf"""
    \documentclass{{article}}
    \usepackage[margin=1in]{{geometry}}
    \usepackage{{booktabs}}
    \usepackage{{caption}}

    \begin{{document}}

    \begin{{table}}[h!]
    \centering
    {table_latex}
    \caption{{{caption}}}
    \label{{{label}}}
    \end{{table}}

    \end{{document}}
    """

    # Save to .tex file
    with open(output_tex_path, 'w') as f:
        f.write(document)

    print(f"LaTeX document saved to: {output_tex_path}")

def csv_to_heatmap(csv_path, output_png_path):
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Define bin width
    bin_width = 200000
    min_size = df['Size'].min()
    max_size = df['Size'].max()
    bins = np.arange(min_size, max_size + bin_width, bin_width)

    # Assign Size bins
    df['SizeBin'] = pd.cut(df['Size'], bins=bins, include_lowest=True)

    # Map bins to mean Size
    mean_labels = {
        b: int(df[df['SizeBin'] == b]['Size'].mean())
        for b in df['SizeBin'].unique()
    }
    df['SizeMean'] = df['SizeBin'].map(mean_labels)

    # Melt to long format
    df_long = pd.melt(df, id_vars=['Members', 'SizeMean'], value_vars=['Pre', 'OT'],
                    var_name='Metric', value_name='Value')

    # Create column label: e.g., "500000 - Pre"
    df_long['Column'] = df_long['SizeMean'].astype(str) + ' - ' + df_long['Metric']

    # Pivot to wide format
    heatmap_data = df_long.pivot_table(index='Members', columns='Column', values='Value')

    # --- Sort columns by Size first, then Metric ---
    # Extract size and metric from column labels
    col_split = heatmap_data.columns.str.extract(r"(\d+)\s*-\s*(Pre|OT)")
    col_split.columns = ['Size', 'Metric']
    col_split['Size'] = col_split['Size'].astype(int)
    col_split['MetricOrder'] = col_split['Metric'].map({'Pre': 0, 'OT': 1})

    # Sort columns by Size, then Metric
    sorted_columns = heatmap_data.columns[np.lexsort((col_split['MetricOrder'], col_split['Size']))]
    heatmap_data = heatmap_data[sorted_columns]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(heatmap_data, cmap='magma', aspect='auto')

    # Axis labels
    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha="right")
    ax.set_yticklabels(heatmap_data.index)

    # Add values in cells
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            val = heatmap_data.iloc[i, j]
            if not pd.isna(val):
                #ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white")
                # simply for the color
                rgb = plt.cm.magma(val/heatmap_data.max().max())
                
                # Convert the background color to a brightness value and choose white/black for text color
                brightness = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
                text_color = "white" if brightness < 0.66 else "black"
                
                # Add the annotation with the selected text color
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color)


    # Titles
    #ax.set_title("Pre and OT per Size Bin and Member Count")
    #ax.set_xlabel("Size Bin and Metric")
    ax.set_ylabel("Ensemble Members")

    plt.tight_layout()

    # Save to file
    plt.savefig(output_png_path, dpi=300)

def csv_to_lineplot(csv_path, output_png_path):
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Compute total time
    df['Total'] = df['Pre'] + df['OT']

    # Hardcoded axis labels
    x_label = "Input Size"
    y_label = "Computation Time (s)"

    plt.rc('font', size=17)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique member counts and setup colormap
    members_list = sorted(df['Members'].unique())
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.1, 1, len(members_list)))

    # Plot each line
    for i, member in enumerate(members_list):
        subset = df[df['Members'] == member].sort_values(by="Size")
        ax.plot(subset['Size'], subset['Total'], label=f"{member}",
                color=colors[i], marker='o')

    # Labels and legend
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #ax.set_title("Total Time vs. Size for Different Member Counts")
    ax.legend(title="Members:")

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)

def csv_to_lineplot_parameter(csv_path, output_png_path):
    from matplotlib import ticker

    # Load the CSV
    df = pd.read_csv(csv_path)

    # Compute total time (choose what you want: Pre, OT, or both)
    #df['Total'] = df['OT']
    #df['Total'] = df['Pre']
    df['Total'] = df['Pre'] + df['OT']

    # Hardcoded axis labels
    x_label = "Input Size"
    y_label = "Computation Time (s)"

    plt.rc('font', size=17)

    # Unique values
    selected_outer = 'Param'
    selected_inner = 'Members'

    members_list = sorted(df[selected_outer].unique())
    params_list = sorted(df[selected_inner].unique())
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.1, 1, len(params_list)))

    # One plot per member value
    for i, member in enumerate(members_list):
        fig, ax = plt.subplots(figsize=(10, 6))  # ← Reset here per plot

        for j, param in enumerate(params_list):
            subset = df[(df[selected_inner] == param) & (df[selected_outer] == member)].sort_values(by="Size")
            if not subset.empty:
                ax.plot(subset['Size'], subset['Total'], label=f"{param}",
                        color=colors[j], marker='o')

        # Labels and legend
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(title="Members:")
        #ax.set_title(f"Computation Time for Members = {member}")

        plt.tight_layout()

        # Output path per member
        path = pathlib.Path(output_png_path)
        out_path = path.with_name(f"{path.stem}_{member}_members{path.suffix}")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)  # ← Important to avoid memory issues

def csv_to_lineplot_size(csv_path, output_png_path):

    from matplotlib import ticker
    
    df = pd.read_csv(csv_path)

    # Group by Size and compute mean
    grouped = df.groupby("Size").mean(numeric_only=True).reset_index()

    plt.rc('font', size=17)

    # Setup plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Get the default matplotlib color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    greens = plt.get_cmap("Greens")(np.linspace(0.5, 0.9, 2))
    purples = plt.get_cmap("Purples")(np.linspace(0.5, 0.8, 2))
    color_pre = greens[0]   # blue
    color_ot = greens[1]    # orange
    color_depth = purples[1] # red


    ax1.set_xlabel("Input Size")
    ax1.set_ylabel("Computation Time (s)", color=color_ot)
    ax1.plot(grouped['Size'], grouped['Pre'], label='Pre', color=color_pre, marker='s')
    ax1.plot(grouped['Size'], grouped['OT'], label='OT', color=color_ot, marker='s')
    ax1.tick_params(axis='y', labelcolor=color_ot)

    # Create a second y-axis for Depth
    ax2 = ax1.twinx()
    ax2.set_ylabel("Avg. Octree Depth", color=color_depth)
    ax2.plot(grouped['Size'], grouped['Depth'], label='Depth', color=color_depth, marker='o')
    ax2.tick_params(axis='y', labelcolor=color_depth)

    # Title and legend
    #fig.suptitle("Pre & OT Time with Depth vs. Input Size")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)

def csv_to_just_two_lines(csv_path, output_png_path):
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Group by Size to get mean values across all members
    grouped = df.groupby("Size").mean(numeric_only=True).reset_index()

    plt.rc('font', size=17)

    # Hardcoded axis labels
    x_label = "Input Size"
    y_label = "Computation Time (s)"

    # Prepare plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get all unique truncation values
    truncation_values = sorted(df["Truncate"].unique())

    # Create color gradients for Pre (reds) and OT (blues)
    blues = plt.get_cmap("Greens")(np.linspace(0.4, 0.9, len(truncation_values)))
    reds = plt.get_cmap("Purples")(np.linspace(0.4, 0.9, len(truncation_values)))

    for i, trunc in enumerate(reversed(truncation_values)):
        subset = df[df["Truncate"] == trunc].sort_values("Size")

        # Plot OT
        ax.plot(subset["Size"], subset["Naive"], label=f"{trunc}",
                color=reds[-i + 2], marker='o')

    for i, trunc in enumerate(reversed(truncation_values)):
        subset = df[df["Truncate"] == trunc].sort_values("Size")

        # Plot Pre
        ax.plot(subset["Size"], subset["Oct"], label=f"{trunc}",
                color=blues[-i + 2], marker='s')

    # Set log scale for y-axis
    ax.set_yscale("log")

    trunc = truncation_values[0]  # First one for example
    subset = df[df["Truncate"] == trunc].sort_values("Size")

    # Sample two points
    sample1 = subset.iloc[-1]  # third point
    #sample2 = subset.iloc[-1] # last point

    # Annotate the "Pre" value
    ax.annotate(f"55.5s",
                xy=(sample1['Size'], sample1['Naive']),
                xytext=(-25, -40),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='-'),
                fontsize=17
    )

    # Annotate the "OT" value
    ax.annotate(f"3.2s",
                xy=(sample1['Size'], sample1['Oct']),
                xytext=(-25, -40),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='-'),
                fontsize=17
    )


    # Add labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #ax.set_title("Pre vs. OT Time by Input Size")
    ax.legend(title="Truncate:")

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)

def cubic_function_plot(output_png_path):

    plt.rc('font', size=17)

    # Create x values from 0 to 1
    x = np.linspace(0, 1, 100)
    # Apply cubic mapping
    y = [easeInOutCubic(x_i) for x_i in x]

    y = np.linspace(0, 1, 100)

    circle_x = np.linspace(0, 1, 7)
    circle_y = np.linspace(0, 1, 7)
    #circle_y = [easeInOutCubic(xi) for xi in circle_x]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 1.7))  # 1:5 ratio (height:width ≈ 2:10)

    ax.plot(x, y, color='black', linewidth=2)
    #ax.set_xlabel('Input')
    #ax.set_ylabel('Output')
    #ax.grid(True)

    # Plot the circles
    ax.scatter(circle_x, circle_y, color='black', s=50, zorder=5)

    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylim(0 - 0.1, 1 + 0.1)

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)


def hausdorff_and_chamfer_distance_fast(A, B):
    """
    Computes the Hausdorff and the Chamfer distance between two sets of 3D points using KDTree for efficiency.
    """
    tree_A = KDTree(A)
    tree_B = KDTree(B)

    # Hausdorff distance
    d_AB = max(tree_B.query(a, k=1)[0] for a in A)
    d_BA = max(tree_A.query(b, k=1)[0] for b in B)
    hd = max(d_AB, d_BA)
    print(f"Hausdorff distance {hd}")

    # Chamfer Distance 
    d_AB = np.mean(tree_B.query(A, k=1)[0])
    d_BA = np.mean(tree_A.query(B, k=1)[0])
    cd = d_AB + d_BA
    print(f"Chamfer Distance: {cd}")
