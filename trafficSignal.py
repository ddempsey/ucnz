# Traffic-signal MDP in one class
import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import (
    Dropdown, FloatSlider, RadioButtons, IntSlider,
    Button, VBox, HBox, Output
)
from IPython.display import display

class TrafficSignalMDP:
    """Single-intersection toy simulator with user-controlled cycle lengths.

    NEW  (v3 – 2025-05-15)
    ─────────────────────────────────────────────────────────────────────────
    • Vehicles that leave the junction now re-appear on the **opposite side
      but with the same lateral offset** as their entry lane.
    • Each change of phase inserts a **2-second yellow interval** in which
      no traffic is discharged and the previously-green lanes are shown in
      yellow.
    """

    ACTIONS = {0: "NS-green", 1: "EW-green"}         # 0 = NS, 1 = EW

    # ─────────────────────────── construction ──────────────────────────────
    def __init__(self, horizon=300, mu_service=0.9):
        # traffic state ------------------------------------------------------
        self.horizon = horizon
        self.mu = mu_service
        self.queues = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        self.phase = 0              # start with NS green
        self.time_in_phase = 0      # s already spent in current (green) phase
        self.in_yellow = False      # ⇠ NEW
        self.yellow_elapsed = 0     # ⇠ NEW: how many yellow seconds so far
        self.t = 0
        self.total_reward = 0.0
        self.just_served = {k: 0 for k in self.queues}

        # figure -------------------------------------------------------------
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        plt.close(self.fig)
        self.fig_out = Output()
        with self.fig_out:
            display(self.fig)

        # widgets ------------------------------------------------------------
        self.dur_NS = IntSlider(10, 1, 60, 1, description='NS green s')
        self.dur_EW = IntSlider(10, 1, 60, 1, description='EW green s')

        self.lam_NS = FloatSlider(value=0.30, min=0, max=1, step=0.05,
                                  description='λ NS veh/s')
        self.lam_EW = FloatSlider(value=0.30, min=0, max=1, step=0.05,
                                  description='λ EW veh/s')

        self.phase_radio = RadioButtons(
            options=[('NS-green', 0), ('EW-green', 1)],
            description='Current'
        )

        self.step_btn = Button(description='Step (1 s)')
        self.step20_btn = Button(description='Step ×20')
        self.reset_btn = Button(description='Reset')

        # callbacks ----------------------------------------------------------
        self.step_btn.on_click(lambda _: self.simulate(1))
        self.step20_btn.on_click(lambda _: self.simulate(20))
        self.reset_btn.on_click(self.on_reset)

        self.phase_radio.observe(
            lambda ch: setattr(self, "phase", ch["new"]), names="value"
        )

        # layout -------------------------------------------------------------
        controls = VBox([
            self.dur_NS, self.dur_EW,
            self.lam_NS, self.lam_EW,
            self.phase_radio,
            HBox([self.step_btn, self.step20_btn]),
            self.reset_btn,
        ])
        display(VBox([HBox([self.fig_out, controls])]))

        self.update_display()

    # ─────────────────────── core simulation ───────────────────────────────
    def simulate(self, n_seconds):
        """Run *n_seconds* of 1-s ticks, switching phase automatically when
        the programmed green time elapses (with a 2 s yellow clearance)."""
        for _ in range(n_seconds):
            if self.t >= self.horizon:
                break
            reward = self._tick_once()
            self.total_reward += reward
            self.t += 1
        self.update_display()

    def _tick_once(self):
        """One-second environment tick."""
        # reset log of departures
        self.just_served = {k: 0 for k in self.queues}

        # 1) Poisson arrivals – independent λ for NS and EW approaches
        for k in ['N', 'S']:
            self.queues[k] += np.random.poisson(self.lam_NS.value)
        for k in ['E', 'W']:
            self.queues[k] += np.random.poisson(self.lam_EW.value)

        # 2) Departures (only if not in yellow)
        green = ['N', 'S'] if self.phase == 0 else ['E', 'W']
        if not self.in_yellow:
            for k in green:
                served = min(self.queues[k], 2)        # 2 veh/s cap
                self.queues[k] -= served
                self.just_served[k] = served

        # 3) Reward (negative queue; 2 s lost-time penalty when yellow starts)
        reward = -sum(self.queues.values())

        # 4) Timing / phase logic -------------------------------------------
        if self.in_yellow:
            # currently in yellow; count its length then flip phase
            self.yellow_elapsed += 1
            if self.yellow_elapsed >= 2:               # 2 s yellow done
                self.in_yellow = False
                self.phase = 1 - self.phase            # actual switch now
                self.phase_radio.value = self.phase
                self.time_in_phase = 0                 # reset green timer
        else:
            # currently green; advance its timer
            self.time_in_phase += 1
            dur_target = (self.dur_NS.value if self.phase == 0
                          else self.dur_EW.value)
            if self.time_in_phase >= dur_target:
                # begin 2 s yellow clearance
                self.in_yellow = True
                self.yellow_elapsed = 0
                reward -= 2.0                          # lost-time penalty

        return reward

    # ─────────────────────────── drawing ────────────────────────────────────
    def draw(self):
        ax = self.ax
        ax.clear()
        ax.set_xlim(0, 5); ax.set_ylim(0, 5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
        ax.set_facecolor('white')
        ax.add_patch(plt.Rectangle((2, 2), 1, 1, fc='lightgrey', ec='k'))

        # queue bars ---------------------------------------------------------
        lane_geom = dict(
            N=((2.1, 3.0),  0,  1),
            S=((2.6, 2.0),  0, -1),
            E=((3.0, 2.1),  1,  0),
            W=((2.0, 2.6), -1,  0)
        )
        max_q = max(1, max(self.queues.values()))
        green_now = ['N', 'S'] if self.phase == 0 else ['E', 'W']
        for k, ((x0, y0), dx, dy) in lane_geom.items():
            # colour logic ---------------------------------------------------
            if self.in_yellow and k in green_now:
                fc = 'gold'               # yellow clearance
            elif (self.phase == 0 and k in ['N', 'S']) or \
                 (self.phase == 1 and k in ['E', 'W']):
                fc = 'tab:green'          # current green phase
            else:
                fc = 'tab:red'            # red

            # draw queue bar --------------------------------------------------
            q = self.queues[k]
            length = 1.5 * (q / max_q)
            ax.add_patch(plt.Rectangle(
                (x0, y0),
                dx * length if dx else 0.3,
                dy * length if dy else 0.3,
                fc=fc, ec='k'))
            text_x = x0 + (dx * length) / 2 + (0.15 if dx == 0 else 0)
            text_y = y0 + (dy * length) / 2 + (0.15 if dy == 0 else 0)
            ax.text(text_x, text_y, str(q),
                    ha='center', va='center', fontsize=10, color='w')

        # departing vehicles -------------------------------------------------
        # (nothing drawn during yellow → just_served all zeros)
        xc, yc = 2.5, 2.5                        # intersection centre
        for k, served in self.just_served.items():
            if served == 0:
                continue
            (x0, y0), dx_q, dy_q = lane_geom[k]
            dx_move, dy_move = -dx_q, -dy_q      # direction of travel

            # start just inside the junction
            start = np.array([xc + 0.15 * dx_move,
                              yc + 0.15 * dy_move])
            # shift laterally so the exit lane has the same offset
            if dx_move == 0:                     # N/S traffic → adjust x
                start[0] += (x0 - xc) + 0.15
            else:                               # E/W traffic → adjust y
                start[1] += (y0 - yc) + 0.15

            # end point a little downstream of the centre
            end = start + np.array([dx_move, dy_move]) * 0.6

            # open green square at end point
            ax.add_patch(plt.Rectangle(
                end - 0.15, 0.3, 0.3,
                fill=False, lw=1.8, ec='green'))

            # arrow from start → end
            ax.annotate("",
                        xy=end,
                        xytext=start,
                        arrowprops=dict(arrowstyle="->",
                                        lw=1.5,
                                        color='green'))

        # headline -----------------------------------------------------------
        phase_name = 'NS' if self.phase == 0 else 'EW'
        bar = 'yellow' if self.in_yellow else 'green'
        ax.set_title(
            f"t = {self.t}s   "#phase = {phase_name} ({bar})   "
            f"time in phase = {self.time_in_phase}s   "
            f"\nΣQ = {sum(self.queues.values())}   "
            f"reward cum = {self.total_reward:.1f}"
        )

    # ───────────────────────── GUI update ───────────────────────────────────
    def update_display(self):
        self.draw()
        self.fig_out.clear_output(wait=True)
        with self.fig_out:
            display(self.fig)

    # ─────────────────────────── reset ──────────────────────────────────────
    def on_reset(self, _btn):
        self.queues = {k: 0 for k in self.queues}
        self.just_served = {k: 0 for k in self.just_served}
        self.phase = 0
        self.time_in_phase = 0
        self.in_yellow = False
        self.yellow_elapsed = 0
        self.t = 0
        self.total_reward = 0.0
        self.update_display()
