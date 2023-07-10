from verse.plotter.plotter2D import *
from fixed_wing_agent import FixedWingAgent
from verse import Scenario, ScenarioConfig
from enum import Enum, auto
import copy
import os 

class FixedWingMode(Enum):
    # NOTE: Any model should have at least one mode
    Normal = auto()
    # TODO: The one mode of this automation is called "Normal" and auto assigns it an integer value.
    # Ultimately for simple models we would like to write
    # E.g., Mode = makeMode(Normal, bounce,...)


# class TrackMode(Enum):
#     Lane0 = auto()
#     #For now this is a dummy notion of Lane


class State:
    """Defines the state variables of the model
    Both discrete and continuous variables
    """

    x: float
    y: float 
    z: float 
    roll: float  
    pitch: float 
    yaw: float 
    x_ref: float 
    y_ref: float 
    z_ref: float 
    mode: FixedWingMode

    def __init__(self, x, y, z, roll, pitch, yaw, x_ref, y_ref, z_ref, fixed_wing_mode: FixedWingMode):
        pass


def decisionLogic(ego: State):
    """Computes the possible mode transitions"""
    output = copy.deepcopy(ego)
    return output


if __name__ == "__main__":
    fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False)) 
    script_path = os.path.realpath(os.path.dirname(__file__))
    fixed_wing_controller = os.path.join(script_path, __file__)
    aircraft = FixedWingAgent("a1", file_name=fixed_wing_controller)
    fixed_wing_scenario.add_agent(aircraft)
    fixed_wing_scenario.set_init(
        [
            [[-2550.0, -20, 110.0, 0, -np.deg2rad(3), 0, -2500.0, 0, 120.0], [-2510.0, 20, 130.0, 0, -np.deg2rad(3), 0, -2500.0, 0, 120.0]]
        ],
        [
            (FixedWingMode.Normal,)
        ],
    )
    # TODO: WE should be able to initialize each of the balls separately
    # this may be the cause for the VisibleDeprecationWarning
    # TODO: Longer term: We should initialize by writing expressions like "-2 \leq myball1.x \leq 5"
    # "-2 \leq myball1.x + myball2.x \leq 5"
    # traces = fixed_wing_scenario.verify(15, 0.05)
    # # TODO: There should be a print({traces}) function
    # fig1 = go.Figure()
    # fig1 = reachtube_tree(traces, None, fig1, 0, 1, [1, 2], "fill", "trace")
    # fig1.show()
    # fig2 = go.Figure()
    # fig2 = reachtube_tree(traces, None, fig2, 0, 2, [1, 2], "fill", "trace")
    # fig2.show()
    # fig3 = go.Figure()
    # fig3 = reachtube_tree(traces, None, fig3, 0, 3, [1, 2], "fill", "trace")
    # fig3.show()

    fig = go.Figure()
    for i in range(10):
        traces = fixed_wing_scenario.simulate(15, 0.05)
        fig = simulation_tree(traces, None, fig, 0, 1)
    fig.show()