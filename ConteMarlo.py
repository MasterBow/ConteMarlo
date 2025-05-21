import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ================================
# 1. Configuración y utilidades
# ================================
DEFAULT_NUM_RUNS = 500
DEFAULT_LAMBDA = 1.0  # tasa de arribos (λ)
DEFAULT_MU = 1.2      # tasa de servicio (μ)
DEFAULT_SIM_TIME = 100.0  # tiempo de simulación
DEFAULT_SERVERS = 1        # número de servidores (c)
RNG_SEED = 42
np.random.seed(RNG_SEED)

# ================================
# 2. Recolector de estadísticas
# ================================
class StatsCollector:
    def __init__(self):
        self.waits = []

    def add_wait(self, wait_time):
        self.waits.append(wait_time)

    def results(self):
        arr = np.array(self.waits)
        return {
            'mean_wait': arr.mean() if len(arr) else 0,
            'var_wait': arr.var(ddof=1) if len(arr) > 1 else 0,
            'max_wait': arr.max() if len(arr) else 0,
            'count': len(arr)
        }

# ================================
# 3. Modelo de colas M/M/c
# ================================
class QueueSystem:
    def __init__(self, arrival_rate, service_rate, servers, sim_time):
        self.lambda_ = arrival_rate
        self.mu = service_rate
        self.c = max(1, servers)
        self.sim_time = sim_time
        self.clock = 0.0
        self.busy_servers = 0
        self.queue = []
        self.stats = StatsCollector()

    def run(self):
        t_next_arrival = np.random.exponential(1/self.lambda_)
        t_next_departure = []  # lista de tiempos de salida por servidor

        while self.clock < self.sim_time:
            next_depart = min(t_next_departure) if t_next_departure else float('inf')
            if t_next_arrival <= next_depart:
                self.clock = t_next_arrival
                if self.busy_servers < self.c:
                    self.busy_servers += 1
                    t_next_departure.append(self.clock + np.random.exponential(1/self.mu))
                else:
                    self.queue.append(self.clock)
                t_next_arrival = self.clock + np.random.exponential(1/self.lambda_)
            else:
                self.clock = next_depart
                idx = t_next_departure.index(next_depart)
                t_next_departure.pop(idx)
                if self.queue:
                    t_arrival = self.queue.pop(0)
                    wait_time = self.clock - t_arrival
                    self.stats.add_wait(wait_time)
                    t_next_departure.append(self.clock + np.random.exponential(1/self.mu))
                else:
                    self.busy_servers -= 1
        return self.stats.results()

# ================================
# 4. Simulación Monte Carlo
# ================================
class MonteCarloRunner:
    def __init__(self, model_cls, runs, params, progress_queue=None):
        self.model_cls = model_cls
        self.runs = runs
        self.params = params
        self.means = []
        self.progress_queue = progress_queue

    def run_all(self):
        self.means.clear()
        for i in range(1, self.runs + 1):
            res = self.model_cls(**self.params).run()
            self.means.append(res['mean_wait'])
            if self.progress_queue:
                self.progress_queue.put(i)
        return self.means

# ================================
# 5. Controlador y GUI con threading, barra de progreso y opciones
# ================================
class AppController:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulación Teoría de Colas y Monte Carlo")
        self.build_controls()

    def build_controls(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0, sticky='nsew')

        # Parámetros
        labels = ["Tasa llegada (λ):", "Tasa servicio (μ):", "Tiempo sim.:", "Corridas:", "Servidores (c):"]
        defaults = [DEFAULT_LAMBDA, DEFAULT_MU, DEFAULT_SIM_TIME, DEFAULT_NUM_RUNS, DEFAULT_SERVERS]
        self.entries = []
        for i, (lbl, val) in enumerate(zip(labels, defaults)):
            ttk.Label(frm, text=lbl).grid(row=i, column=0, sticky='w')
            ent = ttk.Entry(frm); ent.insert(0, str(val)); ent.grid(row=i, column=1)
            self.entries.append(ent)

        # Botones
        self.btn_run = ttk.Button(frm, text="Ejecutar", command=self.start_sim)
        self.btn_run.grid(row=5, column=0, pady=5)
        self.btn_reset = ttk.Button(frm, text="Limpiar", command=self.reset)
        self.btn_reset.grid(row=5, column=1)

        # Barra de progreso
        self.progress = ttk.Progressbar(frm, orient='horizontal', length=200, mode='determinate')
        self.progress.grid(row=6, column=0, columnspan=2, pady=5)

        # Canvas gráfico
        self.fig = plt.Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=1, column=0)

    def start_sim(self):
        try:
            lam = float(self.entries[0].get())
            mu = float(self.entries[1].get())
            sim_time = float(self.entries[2].get())
            runs = int(self.entries[3].get())
            servers = int(self.entries[4].get())
        except ValueError:
            messagebox.showerror("Error", "Parameters inválidos")
            return

        self.btn_run.config(state='disabled')
        self.progress['maximum'] = runs
        self.progress['value'] = 0

        self.progress_queue = queue.Queue()
        thread = threading.Thread(
            target=self.run_thread,
            args=(lam, mu, servers, sim_time, runs, self.progress_queue),
            daemon=True
        )
        thread.start()
        self.root.after(100, self.check_progress)

    def run_thread(self, lam, mu, servers, sim_time, runs, progress_q):
        params = {'arrival_rate': lam, 'service_rate': mu, 'servers': servers, 'sim_time': sim_time}
        runner = MonteCarloRunner(QueueSystem, runs, params, progress_q)
        means = runner.run_all()
        self.plot_results(means)
        self.btn_run.config(state='normal')

    def check_progress(self):
        try:
            while True:
                self.progress['value'] = self.progress_queue.get_nowait()
        except queue.Empty:
            pass
        if self.btn_run['state'] == 'disabled':
            self.root.after(100, self.check_progress)

    def plot_results(self, data):
        self.ax.clear()
        self.ax.hist(data, bins=20)
        self.ax.set_title("Histograma de tiempo medio de espera")
        self.ax.set_xlabel("Tiempo medio de espera")
        self.ax.set_ylabel("Frecuencia")
        self.canvas.draw()

    def reset(self):
        defaults = [DEFAULT_LAMBDA, DEFAULT_MU, DEFAULT_SIM_TIME, DEFAULT_NUM_RUNS, DEFAULT_SERVERS]
        for ent, val in zip(self.entries, defaults):
            ent.delete(0, tk.END)
            ent.insert(0, str(val))
        self.ax.clear(); self.canvas.draw()

# ================================
# 6. Entrada principal
# ================================
def main():
    root = tk.Tk()
    AppController(root)
    root.mainloop()

if __name__ == '__main__':
    main()
