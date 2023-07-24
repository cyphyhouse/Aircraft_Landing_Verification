from verse.plotter.plotter2D import *
from fixed_wing_agent import FixedWingAgent
from fixed_wing_agent2 import AircraftTrackingAgent
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
    yaw: float 
    pitch: float 
    v: float
    x_ref: float 
    y_ref: float 
    z_ref: float 
    yaw_ref: float 
    pitch_ref: float 
    v_ref: float
    mode: FixedWingMode

    def __init__(self, x, y, z, yaw, pitch, v, x_ref, y_ref, z_ref, yaw_ref, pitch_ref, v_ref, mode: FixedWingMode):
        pass


def decisionLogic(ego: State):
    """Computes the possible mode transitions"""
    output = copy.deepcopy(ego)
    return output


if __name__ == "__main__":
    fixed_wing_scenario = Scenario(ScenarioConfig(parallel=False)) 
    script_path = os.path.realpath(os.path.dirname(__file__))
    fixed_wing_controller = os.path.join(script_path, 'fixed_wing2.py')
    aircraft = AircraftTrackingAgent("a1", file_name=fixed_wing_controller)
    fixed_wing_scenario.add_agent(aircraft)
    # x, y, z, yaw, pitch, v
    fixed_wing_scenario.set_init(
        [[
            [-3050.0, -20, 110.0, 0-0.0001, -np.deg2rad(3)-0.0001, 10-0.0001, -3000.0, 0, 120.0, 0, -np.deg2rad(3), 10], 
            [-3010.0, 20, 130.0, 0+0.0001, -np.deg2rad(3)+0.0001, 10+0.0001, -3000.0, 0, 120.0, 0, -np.deg2rad(3), 10]
        ]],
        [
            (FixedWingMode.Normal,)
        ],
    )
    # TODO: WE should be able to initialize each of the balls separately
    # this may be the cause for the VisibleDeprecationWarning
    # TODO: Longer term: We should initialize by writing expressions like "-2 \leq myball1.x \leq 5"
    # "-2 \leq myball1.x + myball2.x \leq 5"
    traces = fixed_wing_scenario.verify(100, 0.1)
    # TODO: There should be a print({traces}) function
    fig1 = go.Figure()
    fig1 = reachtube_tree(traces, None, fig1, 0, 1, [1, 2], "fill", "trace")
    fig1.show()
    fig2 = go.Figure()
    fig2 = reachtube_tree(traces, None, fig2, 0, 2, [1, 2], "fill", "trace")
    fig2.show()
    fig3 = go.Figure()
    fig3 = reachtube_tree(traces, None, fig3, 0, 3, [1, 2], "fill", "trace")
    fig3.show()
    fig4 = go.Figure()
    fig4 = reachtube_tree(traces, None, fig4, 0, 4, [1, 2], "fill", "trace")
    fig4.show()
    fig5 = go.Figure()
    fig5 = reachtube_tree(traces, None, fig5, 0, 5, [1, 2], "fill", "trace")
    fig5.show()
    fig6 = go.Figure()
    fig6 = reachtube_tree(traces, None, fig6, 0, 6, [1, 2], "fill", "trace")
    fig6.show()

    # fig1 = go.Figure()
    # fig2 = go.Figure()
    # fig3 = go.Figure()
    # fig4 = go.Figure()
    # fig5 = go.Figure()
    # fig6 = go.Figure()
    # for i in range(10):
    #     traces = fixed_wing_scenario.simulate(100, 0.05)
    #     fig1 = simulation_tree(traces, None, fig1, 0, 1)
    #     fig2 = simulation_tree(traces, None, fig2, 0, 2)
    #     fig3 = simulation_tree(traces, None, fig3, 0, 3)
    #     fig4 = simulation_tree(traces, None, fig4, 0, 4)
    #     fig5 = simulation_tree(traces, None, fig5, 0, 5)
    #     fig6 = simulation_tree(traces, None, fig6, 0, 6)
    # fig1.show()
    # fig2.show()
    # fig3.show()
    # fig4.show()
    # fig5.show()
    # fig6.show()
