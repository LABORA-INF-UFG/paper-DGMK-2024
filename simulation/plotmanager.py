import os
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from simulation.simulation import Simulation
from simulation.slice import Slice
from simulation.user import User

def mean(l: List[float]) -> float:
    return sum(l)/len(l)

class PlotManager:
    
    def __init__(
        self,
        sim_names:List[str],
        seeds:List[int],
        slice_names:Dict[int, str],
        colors:Dict[str,str],
        metrics_folder: str = "metrics",
        plots_folder:str = "plots",
        rename:Dict[str,str] = {}
    ) -> None:
        self.slice_name = slice_names
        self.colors = colors
        self.metrics_folder = metrics_folder
        self.plots_folder = plots_folder
        self.rename = rename
        self.data_description:Dict[int, Dict[str: Dict]] = {seed: {} for seed in seeds}
        self.simulation_data:Dict[int, Dict[str: pd.DataFrame]] = {seed: {} for seed in seeds}
        self.slice_data:Dict[int, Dict[str: Dict[int, pd.DataFrame]]] = {seed: {} for seed in seeds}
        self.user_data:Dict[int, Dict[str: Dict[int, pd.DataFrame]]] = {seed: {} for seed in seeds}

        # Creating folders
        if not os.path.exists(self.plots_folder):
            os.makedirs(self.plots_folder)
        for seed in seeds:
            if not os.path.exists(self.plots_folder+"/"+str(seed)):
                os.makedirs(self.plots_folder+"/"+str(seed))
            for sim_name in sim_names:
                if not os.path.exists(self.plots_folder+"/"+str(seed)+"/"+sim_name+"/"):
                    os.makedirs(self.plots_folder+"/"+str(seed)+"/"+sim_name+"/")
            for slice_id in slice_names.keys():
                if not os.path.exists(self.plots_folder+"/"+str(seed)+"/"+slice_names[slice_id]+"/"):
                    os.makedirs(self.plots_folder+"/"+str(seed)+"/"+slice_names[slice_id]+"/")
        
        # Loading data
        for seed in seeds:
            for sim_name in sim_names:
                self.load_data(sim_name=sim_name, seed=seed)
        
        # Globally setting the font
        plt.rcParams.update({'font.family': 'Times New Roman'})
    
    # Reading all metrics from the simulation
    def load_data(self, sim_name:str, seed:int):
        
        # Reading the simulation description (slices, users, etc.)
        with open(f"{self.metrics_folder}/{sim_name}_{seed}/description.json") as f:
            self.data_description[seed][sim_name] = pd.read_json(f)
        
        # Reading the simulation metrics
        self.simulation_data[seed][sim_name] = pd.read_csv(f"{self.metrics_folder}/{sim_name}_{seed}/sim_metrics.csv")

        # Reading the slice metrics
        self.slice_data[seed][sim_name] = {}
        for slice_id in self.data_description[seed][sim_name]["slices"].keys():
            self.slice_data[seed][sim_name][slice_id] = pd.read_csv(f"{self.metrics_folder}/{sim_name}_{seed}/slice_{slice_id}_metrics.csv")

        # Reading the user metrics
        self.user_data[seed][sim_name] = {}
        for slice_id in self.data_description[seed][sim_name]["slices"].keys():
            for user_id in self.data_description[seed][sim_name]["slices"][slice_id]["users"]:
                self.user_data[seed][sim_name][user_id] = pd.read_csv(f"{self.metrics_folder}/{sim_name}_{seed}/user_{user_id}_metrics.csv")
        
        # Converting capacity values to Mbps
        for metrics in self.simulation_data[seed][sim_name].columns:
            if "capacity" in metrics and "fair" not in metrics:
                self.simulation_data[seed][sim_name][metrics] = self.simulation_data[seed][sim_name][metrics]/1e6
        for slice_id in self.slice_data[seed][sim_name].keys():
            for metrics in self.slice_data[seed][sim_name][slice_id].columns:
                if "capacity" in metrics and "fair" not in metrics:
                    self.slice_data[seed][sim_name][slice_id][metrics] = self.slice_data[seed][sim_name][slice_id][metrics]/1e6
        for user_id in self.user_data[seed][sim_name].keys():
            for metrics in self.user_data[seed][sim_name][user_id].columns:
                if "capacity" in metrics and "fair" not in metrics:
                    self.user_data[seed][sim_name][user_id][metrics] = self.user_data[seed][sim_name][user_id][metrics]/1e6

    def plot_lines(
        self,
        lines:Dict[str,Tuple[List[float],List[float], str]],
        title: str,
        x_label: str,
        y_label: str,
        filename: str,
        figsize:Tuple[int,int] = (6,4),
        fontsize:int = 20,
        y_as_percentage:bool = False,
        x_as_percentage:bool = False,
        yticks:List[float] = None
    ):
        plt.figure(figsize=figsize)
        plt.rcParams.update({'font.size': fontsize})
        for line in lines:
            if len(lines) == 1:
                plt.plot(lines[line][0], lines[line][1], color = lines[line][2])
            else:
                name = line if line not in self.rename else self.rename[line]
                plt.plot(lines[line][0], lines[line][1], color = lines[line][2], label=name)
        if y_as_percentage:
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        if x_as_percentage:
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
        if yticks is not None:
            plt.yticks(yticks)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if len(lines) > 1:
            plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.plots_folder}/{filename}", bbox_inches='tight')
        print(f"Saved plot {self.plots_folder}/{filename}")
        plt.close()

    def plot_bars(
            self,
            bars:Dict[str, Tuple[str, float, str, float]],
            title:str,
            xlabel:str,
            ylabel:str,
            filename: str,
            figsize:Tuple[int,int] = (6,4),
            fontsize:int = 20,
            y_as_percentage:bool = False,
            rotate_x_ticks:bool = False
        ):
        plt.figure()
        plt.gcf().set_size_inches(figsize[0], figsize[1])  # Set the figure aspect ratio
        plt.rcParams.update({'font.size': fontsize})
        for bar in bars.values():
            if bar[1] == 0:
                continue
            name = bar[0] if bar[0] not in self.rename else self.rename[bar[0]]
            if bar[3] is None:
                plt.bar(name, bar[1], color=bar[2])
                plt.text(name, bar[1], f'{bar[1]:.4f}', ha='center', va='bottom', fontsize=fontsize*0.8)
            else:
                lower_error = bar[3] if bar[1] - bar[3] > 0 else bar[1]
                upper_error = bar[3]
                plt.bar(name, bar[1], color=bar[2], yerr=[[lower_error], [upper_error]], capsize=10, error_kw={'elinewidth': 2, 'capthick': 3})
                plt.text(name, bar[1], f'{bar[1]:.4f}', ha='center', va='bottom', fontsize=fontsize*0.8)
        if y_as_percentage:
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        if rotate_x_ticks:
            plt.xticks(rotation=45)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(f"{self.plots_folder}/{filename}", bbox_inches='tight')
        print(f"Saved plot {self.plots_folder}/{filename}")
        plt.close()
    
    def plot_boxplot(
        self,
        boxplots:Dict[str,Tuple[str,List[float],str]],
        title: str,
        xlabel: str,
        ylabel: str,
        filename: str,
        figsize: Tuple[int, int],
        fontsize: int,
        y_as_percentage:bool,
        rotate_x_ticks:bool,
        yticks:List[float] = None
    ):
        plt.figure()
        plt.gcf().set_size_inches(figsize[0], figsize[1])
        plt.rcParams.update({'font.size': fontsize})
        positions = range(1, len(boxplots) + 1)
        data = [boxplot[1] for boxplot in boxplots.values()]
        colors = [boxplot[2] for boxplot in boxplots.values()]
        box = plt.boxplot(data, patch_artist=True, positions=positions, flierprops=dict(marker='o', markersize=3))
        for median in box['medians']:
            median.set(color='black', linewidth=1.5)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        if y_as_percentage:
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        if yticks is not None:
            plt.yticks(yticks)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        names = [boxplot[0] if boxplot[0] not in self.rename else self.rename[boxplot[0]] for boxplot in boxplots.values()]
        plt.xticks(range(1, len(boxplots) + 1), names)
        if rotate_x_ticks:
            plt.xticks(rotation=40)
        plt.tight_layout()
        plt.savefig(f"{self.plots_folder}/{filename}", bbox_inches='tight')
        print(f"Saved plot {self.plots_folder}/{filename}")

    def moving_average(self, data: List[float], window_size: int) -> List[float]:
        if len(data) <= window_size:
            return data
        return pd.Series(data).rolling(window=window_size).mean().tolist()
    
    def mean_and_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        n = len(data)
        mean = np.mean(data)
        sem = stats.sem(data)  # Standard error of the mean
        margin_of_error = sem * stats.t.ppf((1 + confidence) / 2., n-1)
        return mean, margin_of_error

    def plot_metric_boxplot_all_seeds(
        self,
        sim_names: List[str],
        seeds:List[int],
        metric:str,
        title:str = None,
        xlabel:str = None,
        ylabel:str = None,
        figsize: Tuple[int, int] = (5, 4),
        fontsize:int=20,
        y_as_percentage:bool = False,
        rotate_x_ticks:bool = False,
        yticks:List[float] = None
    ):
        boxplots = {}
        for sim_name in sim_names:
            values = []
            for seed in seeds:
                values.extend(list(self.simulation_data[seed][sim_name][metric]))
            boxplots[sim_name] = (
                sim_name,
                values,
                self.colors[sim_name],
            )
        filename = f"boxplot_{metric}.pdf"
        self.plot_boxplot(boxplots=boxplots, title=title, xlabel=xlabel, ylabel=ylabel, filename=filename, figsize=figsize, fontsize=fontsize, y_as_percentage=y_as_percentage,rotate_x_ticks=rotate_x_ticks,yticks=yticks)

    def plot_metric_bar_all_seeds(
        self, sim_names: List[str],
        seeds:List[int],
        metric:str,
        sum_instead_of_mean:bool = False,
        title:str = None,
        xlabel:str = None,
        ylabel:str = None,
        figsize: Tuple[int, int] = (5, 4),
        fontsize:int=20,
        y_as_percentage:bool = False,
        rotate_x_ticks:bool = False
    ):
        bars = {}
        for sim_name in sim_names:
            values = []
            for seed in seeds:
                if sum_instead_of_mean:
                    values.append(self.simulation_data[seed][sim_name][metric].sum())
                else:
                    values.append(self.simulation_data[seed][sim_name][metric].mean())
            value, conf_interval = self.mean_and_confidence_interval(values)
            bars[sim_name] = (
                sim_name,
                value,
                self.colors[sim_name],
                conf_interval
            )
        filename = f"bar_{metric}.pdf"
        self.plot_bars(bars=bars, title=title, xlabel=xlabel, ylabel=ylabel, filename=filename, figsize=figsize, fontsize=fontsize, y_as_percentage=y_as_percentage, rotate_x_ticks=rotate_x_ticks)

    def plot_cdf_all_seeds(
        self,
        sim_names: List[str],
        seeds:List[int],
        metric:str,
        title:str = None,
        xlabel:str = None,
        ylabel:str = "CDF",
        figsize: Tuple[int, int] = (8, 6),
        fontsize:int=20,
        x_as_percentage:bool = False,
        yticks:List[float] = None,
        ccdf:bool = False
    ):         
        lines = {}
        for sim_name in sim_names:
            data = []
            for seed in seeds:
                data.extend(list(self.simulation_data[seed][sim_name][metric]))
            data = np.array(data)
            x = np.sort(data)
            y = np.arange(1, len(x) + 1) / len(x)
            if ccdf:
                y = 1 - y
                x = x
            lines[sim_name] = (
                x,
                y,
                self.colors[sim_name]
            )
        filename = f"cdf_{metric}.pdf" if ccdf == False else f"ccdf_{metric}.pdf"
        if ccdf:
            ylabel = "CCDF"
        self.plot_lines(lines, title=title, x_label=xlabel, y_label=ylabel, filename=filename, figsize=figsize, fontsize=fontsize, y_as_percentage=True, x_as_percentage=x_as_percentage, yticks=yticks)

    def plot_metric_bar_one_seed(
        self, sim_names: List[str],
        seed:int,
        metric:str,
        sum_instead_of_mean:bool = False,
        title:str = None,
        xlabel:str = None,
        ylabel:str = None,
        figsize: Tuple[int, int] = (5, 4),
        fontsize:int=20,
        y_as_percentage:bool = False,
        rotate_x_ticks:bool = False
    ):
        bars = {}
        for sim_name in sim_names:
            if sum_instead_of_mean:
                value = self.simulation_data[seed][sim_name][metric].sum()
            else:
                value = self.simulation_data[seed][sim_name][metric].mean()
            bars[sim_name] = (
                sim_name,
                value,
                self.colors[sim_name],
                None
            )
        filename = f"{seed}/bar_{metric}.pdf"
        self.plot_bars(bars=bars, title=title, xlabel=xlabel, ylabel=ylabel, filename=filename, figsize=figsize, fontsize=fontsize, y_as_percentage=y_as_percentage, rotate_x_ticks=rotate_x_ticks)

    def plot_metric_line_one_seed(
        self,
        sim_names: List[str],
        seed:int,
        metric:str,
        smoothing_window: int = 10,
        title:str = None,
        xlabel:str = None,
        ylabel:str = None,
        figsize: Tuple[int, int] = (5, 4),
        fontsize=20,
        y_as_percentage:bool = False
    ):
        lines = {}
        for sim_name in sim_names:
            steps = self.simulation_data[seed][sim_name]["step"]
            values = self.simulation_data[seed][sim_name][metric]
            smoothed_values = self.moving_average(values, smoothing_window)
            lines[sim_name] = (steps, smoothed_values, self.colors[sim_name])
        filename = f"{seed}/line_{metric}.pdf" if len(sim_names) > 1 else f"{seed}/{sim_names[0]}/line_{metric}.pdf"
        self.plot_lines(lines=lines, title=title, x_label=xlabel, y_label=ylabel, filename=filename, figsize=figsize, fontsize=fontsize, y_as_percentage=y_as_percentage)
    
    def plot_metric_line_one_seed_one_slice_multi_sim(
        self,
        sim_names: List[str],
        seed:int,
        slice_id:int,
        metric:str,
        smoothing_window: int = 10,
        title:str = None,
        xlabel:str = None,
        ylabel:str = None,
        figsize: Tuple[int, int] = (5, 4),
        fontsize=20,
        y_as_percentage:bool = False
    ):
        lines = {}
        for sim_name in sim_names:
            steps = self.slice_data[seed][sim_name][slice_id]["step"]
            values = self.slice_data[seed][sim_name][slice_id][metric]
            smoothed_values = self.moving_average(values, smoothing_window)
            lines[sim_name] = (steps, smoothed_values, self.colors[sim_name])
        filename = f"{seed}/{self.slice_name[slice_id]}/line_{metric}.pdf"
        self.plot_lines(lines=lines, title=title, x_label=xlabel, y_label=ylabel, filename=filename, figsize=figsize, fontsize=fontsize, y_as_percentage=y_as_percentage)
    
    def plot_metric_line_one_seed_one_sim_multi_slices(
        self,
        sim_name: str,
        seed:int,
        metric:str,
        smoothing_window: int = 10,
        title:str = None,
        xlabel:str = None,
        ylabel:str = None,
        figsize: Tuple[int, int] = (5, 4),
        fontsize=20,
        y_as_percentage:bool = False
    ):
        lines = {}
        for slice_id in self.slice_data[seed][sim_name].keys():
            steps = self.slice_data[seed][sim_name][slice_id]["step"]
            values = self.slice_data[seed][sim_name][slice_id][metric]
            smoothed_values = self.moving_average(values, smoothing_window)
            lines[self.slice_name[slice_id]] = (steps, smoothed_values, self.colors[self.slice_name[slice_id]])
        filename = f"{seed}/{sim_name}/line_slices_{metric}.pdf"
        self.plot_lines(lines=lines, title=title, x_label=xlabel, y_label=ylabel, filename=filename, figsize=figsize, fontsize=fontsize, y_as_percentage=y_as_percentage)

    def plot_metric_line_one_seed_one_sim_multi_users(
        self,
        sim_name: str,
        seed:int,
        metric:str,
        smoothing_window: int = 10,
        title:str = None,
        xlabel:str = None,
        ylabel:str = None,
        figsize: Tuple[int, int] = (5, 4),
        fontsize=20,
        y_as_percentage:bool = False
    ):
        lines = {}
        for user_id in self.user_data[seed][sim_name].keys():
            steps = self.user_data[seed][sim_name][user_id]["step"]
            values = self.user_data[seed][sim_name][user_id][metric]
            smoothed_values = self.moving_average(values, smoothing_window)
            lines[user_id] = (steps, smoothed_values, None)
        filename = f"{seed}/{sim_name}/line_users_{metric}.pdf"
        self.plot_lines(lines=lines, title=title, x_label=xlabel, y_label=ylabel, filename=filename, figsize=figsize, fontsize=fontsize, y_as_percentage=y_as_percentage)
    
    def plot_metric_bar_all_seeds_one_slice(
        self,
        sim_names: List[str],
        seeds:List[int],
        metric:str,
        slice_id:int,
        sum_instead_of_mean:bool = False,
        title:str = None,
        xlabel:str = None,
        ylabel:str = None,
        figsize: Tuple[int, int] = (5, 4),
        fontsize:int=20,
        y_as_percentage:bool = False,
        rotate_x_ticks:bool = False
    ):
        bars = {}
        for sim_name in sim_names:
            values = []
            for seed in seeds:
                if sum_instead_of_mean:
                    values.append(self.slice_data[seed][sim_name][slice_id][metric].sum())
                else:
                    values.append(self.slice_data[seed][sim_name][slice_id][metric].mean())
            value, conf_interval = self.mean_and_confidence_interval(values)
            bars[sim_name] = (
                sim_name,
                value,
                self.colors[sim_name],
                conf_interval
            )
        filename = f"slice_{slice_id}_bar_{metric}.pdf"
        self.plot_bars(bars=bars, title=title, xlabel=xlabel, ylabel=ylabel, filename=filename, figsize=figsize, fontsize=fontsize, y_as_percentage=y_as_percentage, rotate_x_ticks=rotate_x_ticks)

    def plot_metric_bar_all_seeds_one_slice_min(
        self,
        sim_names: List[str],
        seeds:List[int],
        metric:str,
        slice_id:int,
        title:str = None,
        xlabel:str = None,
        ylabel:str = None,
        figsize: Tuple[int, int] = (5, 4),
        fontsize:int=20,
        y_as_percentage:bool = False,
        rotate_x_ticks:bool = False
    ):
        bars = {}
        for sim_name in sim_names:
            values = []
            for seed in seeds:
                values.append(self.slice_data[seed][sim_name][slice_id][metric].min())
            bars[sim_name] = (
                sim_name,
                min(values),
                self.colors[sim_name],
                None
            )
        filename = f"slice_{slice_id}_bar_{metric}_min.pdf"
        self.plot_bars(bars=bars, title=title, xlabel=xlabel, ylabel=ylabel, filename=filename, figsize=figsize, fontsize=fontsize, y_as_percentage=y_as_percentage, rotate_x_ticks=rotate_x_ticks)

    def plot_metric_bar_all_seeds_one_slice_max(
        self,
        sim_names: List[str],
        seeds:List[int],
        metric:str,
        slice_id:int,
        title:str = None,
        xlabel:str = None,
        ylabel:str = None,
        figsize: Tuple[int, int] = (5, 4),
        fontsize:int=20,
        y_as_percentage:bool = False,
        rotate_x_ticks:bool = False
    ):
        bars = {}
        for sim_name in sim_names:
            values = []
            for seed in seeds:
                values.append(self.slice_data[seed][sim_name][slice_id][metric].max())
            bars[sim_name] = (
                sim_name,
                max(values),
                self.colors[sim_name],
                None
            )
        filename = f"slice_{slice_id}_bar_{metric}_max.pdf"
        self.plot_bars(bars=bars, title=title, xlabel=xlabel, ylabel=ylabel, filename=filename, figsize=figsize, fontsize=fontsize, y_as_percentage=y_as_percentage, rotate_x_ticks=rotate_x_ticks)
