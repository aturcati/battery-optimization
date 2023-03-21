import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyomo.environ as pyo
import scipy.optimize as optimize
from pyomo.environ import (
    Any,
    ConcreteModel,
    Constraint,
    Objective,
    Param,
    Set,
    SolverFactory,
    Var,
    minimize,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)


@dataclass
class Battery:
    """
    Battery is a dataclass that holds the battery parameters.
    """

    E_0: float = 0.0
    E_final: float = 0.0
    E_max: float = 0.0
    P_max: float = 0.0
    P_min: float = 0.0


class BaseBatteryOptimizer(object, metaclass=ABCMeta):
    """
    BaseBatteryOptimizer is an abstract class that defines the interface for the battery optimizers.
    """

    def __init__(
        self, battery: Battery, prices: npt.NDArray[np.float64], dt: float = 1.0
    ):
        """
        __init__ initializes the battery optimizer.

        Args:
            battery (Battery): battery object.
            prices (npt.NDArray[np.float64]): Array of electricity prices.
            dt (float, optional): Time unit in hours. Defaults to 1.0.
        """
        self.battery = battery
        self.prices = prices
        self.dt = dt

    @abstractmethod
    def _prepare_problem(self) -> None:
        pass

    @abstractmethod
    def optimize(self) -> Tuple[float, pd.DataFrame]:
        pass


class BatteryOptimizer(BaseBatteryOptimizer):
    """
    BatteryOptimizer is a class that implements the battery optimization problem using scipy.optimize.linprog.
    """

    def _prepare_problem(self):
        """
        _prepare_problem prepares the optimization problem by setting up the constraints and bounds.
        """
        self._cdt = self.prices * self.dt

        # Power constraints (1)
        N = len(self.prices)
        low_bounds = np.ones(N) * self.battery.P_min
        upper_bounds = np.ones(N) * self.battery.P_max
        self._b_bounds = np.stack([low_bounds, upper_bounds], axis=1)

        # Inequality constraints (2) and (3)
        ub_ineq = np.ones(N) * (self.battery.E_max - self.battery.E_0)
        lb_ineq = np.ones(N) * self.battery.E_0
        self._A_ineq = np.concatenate(
            [self.dt * np.tri(N), -self.dt * np.tri(N)], axis=0
        )
        self._b_ineq = np.concatenate([ub_ineq, lb_ineq])

        # Equality constraints (4)
        self._A_eq = self.dt * np.ones((1, N))
        self._b_eq = self.battery.E_final - self.battery.E_0

    def optimize(self) -> optimize.OptimizeResult:
        """
        optimize optimizes the battery problem using scipy.optimize.linprog.

        Returns:
            optimize.OptimizeResult: result of the optimization.
        """
        self._prepare_problem()
        self._opt = optimize.linprog(
            self._cdt,
            self._A_ineq,
            self._b_ineq,
            self._A_eq,
            self._b_eq,
            self._b_bounds,
        )
        if self._opt.success:
            logging.info(f"Optimization successful: {self._opt.message}")
            df = pd.DataFrame(
                {
                    "power": self._opt.x,
                    "capacity": np.append(0.0, np.cumsum(self._opt.x * self.dt))[:-1],
                },
                index=np.arange(len(self.prices)),
            )
            return (self._opt.fun, df)
        else:
            logging.info(f"Optimization NOT successful: {self._opt.message}")
            return (0.0, pd.DataFrame({}))


class PyoBatteryOptimizer(BaseBatteryOptimizer):
    """
    PyoBatteryOptimizer is a class that implements the battery optimization problem using Pyomo.

    """

    def __init__(
        self,
        battery: Battery,
        prices: npt.NDArray[np.float64],
        dt: float = 1.0,
        fee: float = 0.0,
        fee_unit: float = 1.0,
    ):
        """
        __init__ initializes the battery optimizer.

        Args:
            battery (Battery): battery object.
            prices (npt.NDArray[np.float64]): Array of electricity prices (Euro/MWh).
            dt (float, optional): Time unit in hours. Defaults to 1.0.
            fee (float, optional): Fee in Euros for discharging the power constraints. Defaults to 0.0.
            fee_unit (float, optional): Unit of the discharge fee in MWh. Defaults to 1.0.
        """
        super().__init__(battery, prices, dt)
        self.fee = fee
        self.fee_unit = fee_unit

    def _prepare_problem(self):
        """
        _prepare_problem prepares the optimization problem by setting up the constraints and bounds.

        """
        self._model = ConcreteModel()
        self._opt = SolverFactory("glpk")

        self._model.It = Set(initialize=np.arange(len(self.prices)), ordered=True)
        self._model.Price = Param(initialize=self.prices, within=Any)

        # battery variables
        self._model.Capacity = Var(self._model.It, bounds=(0.0, self.battery.E_max))
        self._model.Recharge = Var(
            self._model.It, bounds=(0, self.battery.P_max)
        )  # Constraint (1)
        self._model.Discharge = Var(
            self._model.It, bounds=(0, self.battery.P_max)
        )  # Constraint (1)

        # Defining the battery objective (function to be minimised)
        def maximise_profit(model):
            # Revenues from selling electricity
            rev = sum(self.prices[i] * model.Discharge[i] * self.dt for i in model.It)

            # Costs of buying electricity
            cost = sum(self.prices[i] * model.Recharge[i] * self.dt for i in model.It)

            # Fee for discharging
            fee_units = (model.Discharge[i] * self.dt / self.fee_unit for i in model.It)
            fee = sum(self.fee * pu for pu in fee_units)

            return cost + fee - rev

        # Constraint (2)
        def constraint_recharge(model, i):
            return model.Recharge[i] * self.dt <= (
                self.battery.E_max - model.Capacity[i]
            )

        # Constraint (3)
        def constraint_discharge(model, i):
            return model.Discharge[i] * self.dt <= model.Capacity[i]

        def constraint_capacity(model, i):
            # if first iteration, set the capacity to the initial capacity
            if i == model.It.first():
                return model.Capacity[i] == self.battery.E_0
            else:
                return model.Capacity[i] == (
                    model.Capacity[i - 1]
                    + model.Recharge[i - 1] * self.dt
                    - model.Discharge[i - 1] * self.dt
                )

        # Set constraints and objective
        self._model.capacity_constraint = Constraint(
            self._model.It, rule=constraint_capacity
        )
        self._model.constraint_recharge = Constraint(
            self._model.It, rule=constraint_recharge
        )
        self._model.constraint_discharge = Constraint(
            self._model.It, rule=constraint_discharge
        )

        self._model.objective = Objective(rule=maximise_profit, sense=minimize)

    def _get_results(self) -> pd.DataFrame:
        """
        _get_results gets the results from the Pyomo model.

        Returns:
            pd.DataFrame: DataFrame with the results.
        """
        df = pd.DataFrame(
            {
                "capacity": [self._model.Capacity[i]() for i in self._model.It],
                "recharge": [self._model.Recharge[i]() for i in self._model.It],
                "discharge": [self._model.Discharge[i]() for i in self._model.It],
            },
            index=self._model.It,
        )
        df["power"] = df["recharge"] - df["discharge"]
        return df

    def optimize(self) -> Tuple[float, pd.DataFrame]:
        """
        optimize optimizes the battery problem using Pyomo.

        Returns:
            model: Pyomo model.
        """
        self._prepare_problem()
        res = self._opt.solve(self._model)
        if res.solver.status.value == "ok":
            logging.info(
                f"Optimization successful: {res.solver.termination_condition.value}"
            )
            return (
                self._model.objective(),
                self._get_results(),
            )
        else:
            logging.info(
                f"Optimization NOT successful: {res.solver.termination_condition.value}"
            )
            return (0.0, pd.DataFrame({}))
