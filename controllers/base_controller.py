from abc import ABC, abstractmethod
from typing import Any

class BaseController(ABC):
    """
    Abstract base class for drone controllers
    """

    def __init__(self, n: int, control_frequency: float, mass: float, moment_of_inertia: dict, thrust_to_weight: float, limit_min: list, limit_max: list, gravity: float):
        """
        Initialize the base controller with dynamics and core properties.

        Args:
            n (int) : Number of controllers that will run parallely
            control_frequency (float): Control loop frequency in Hz.
            mass (float): Drone's mass in kg.
            moment_of_inertia (dict): Drone's moment of inertia (Ixx, Iyy, Izz).
            thrust_to_weight (float): Thrust-to-weight ratio of the drone.
            limit_min (list): Minimum control limit
            limit_max (list): Maximum control limit
            gravity (float): Gravity of the environment
        """
        self._n = n

        # System dynamics properties
        self._mass = mass
        self._moment_of_inertia = moment_of_inertia
        self._thrust_to_weight = thrust_to_weight
        self._gravity = gravity

        # Control properties
        self._control_frequency = control_frequency
        self._limit_min = limit_min
        self._limit_max = limit_max
        self._current_state = None
        self._reference_state = None
        self._control_output = None
        self._last_timestamp = 0.0

    ### --- Core Methods --- ###
    
    @abstractmethod
    def set_reference(self, reference_state: Any):
        """Set the desired target state for the drone."""
        self._reference_state = reference_state

    @abstractmethod
    def set_current(self, current_state: Any):
        """Set the current state of the drone."""
        self._current_state = current_state

    @abstractmethod
    def compute_control(self, timestamp: float|None = None):
        """
        Compute control based on the current and target states.

        Args:
            timestamp (float | None): Current time for time-dependent calculations.

        Returns:
            dict: Control outputs (e.g., motor commands, thrust, torques).
        """
        if timestamp is not None:
            self._last_timestamp = timestamp

    @abstractmethod
    def reset(self):
        """Reset the controller's internal state."""
        pass

    ### --- Properties --- ###
    
    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, value):
        if value <= 0:
            raise ValueError("Mass must be positive.")
        self._mass = value

    @property
    def moment_of_inertia(self):
        return self._moment_of_inertia

    @moment_of_inertia.setter
    def moment_of_inertia(self, value):
        if not isinstance(value, dict) or not all(k in value for k in ('Ixx', 'Iyy', 'Izz')):
            raise ValueError("Moment of inertia must be a dictionary with keys: 'Ixx', 'Iyy', 'Izz'.")
        self._moment_of_inertia = value

    @property
    def thrust_to_weight(self):
        return self._thrust_to_weight

    @thrust_to_weight.setter
    def thrust_to_weight(self, value):
        if value <= 0:
            raise ValueError("Thrust-to-weight ratio must be positive.")
        self._thrust_to_weight = value

    @property
    def gravity(self):
        return self._gravity

    @gravity.setter
    def gravity(self, value):
        if value <= 0:
            raise ValueError("Gravity must be positive.")
        self._gravity = value

    @property
    def control_frequency(self):
        return self._control_frequency

    @control_frequency.setter
    def control_frequency(self, value):
        if value <= 0:
            raise ValueError("Control frequency must be positive.")
        self._control_frequency = value

    @property
    def limit_min(self):
        return self._limit_min

    @limit_min.setter
    def limit_min(self, limit):
        if not isinstance(limit, list):
            raise ValueError("Limit must be a list of size output.")
        self._limit_min = limit

    @property
    def limit_max(self):
        return self._limit_max

    @limit_max.setter
    def limit_max(self, limit):
        if not isinstance(limit, list):
            raise ValueError("Limit must be a list of size output.")
        self._limit_max = limit

    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, state):
        if not isinstance(state, list):
            raise ValueError("Current state must be a list.")
        self._current_state = state

    @property
    def reference_state(self):
        return self._reference_state

    @reference_state.setter
    def reference_state(self, state):
        if not isinstance(state, list):
            raise ValueError("Reference state must be a list.")
        self._reference_state = state

    @property
    def control_output(self):
        return self._control_output

    @control_output.setter
    def control_output(self, output):
        if not isinstance(output, list):
            raise ValueError("Control output must be a list.")
        self._control_output = output
