import scipy.interpolate
from env.balloon import Balloon
from env.balloon_env import BalloonEnvironment, BalloonERAEnvironment
from env.ERA_wind_field import WindField, WindVector, SimplexWindNoise
from balloon_learning_environment.env.balloon.standard_atmosphere import JaxAtmosphere

import numpy as np
import jax
import jax.numpy as jnp
import scipy
from functools import partial
import datetime as dt
import time
import copy

def sigmoid(x):
    """Returns a sigmoid function with range [-1, 1] instead of [0,1]."""
    return 2 / (1 + np.exp(-x)) - 1

def inverse_sigmoid(x):
    """Returns the inverse of the custom sigmoid defined above. """
    return np.log((x+1)/(1-x))


def grad_descent_optimizer(initial_plan, dcost_dplan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride, alpha=0.01,max_iters=80, isNormalized=0):
    """Basic gradient descent with just a learning rate and also normalizing the gradient. Doing both the learning rate and normalization seemed to perform worse and take longer. High learning rates had power violations. Slower learning rates performed better than baseline but took more time (almost 4s)."""
    print(f"USING LEARNING RATE {alpha}, MAXITERS {max_iters}, ISNORMALIZED {isNormalized}")
    plan = initial_plan

    for gradient_steps in range(max_iters):
        gradient_plan = dcost_dplan(plan, balloon, forecast, atmosphere, terminal_cost_fn, time_delta, stride)

        if  np.isnan(gradient_plan).any() or abs(jnp.linalg.norm(gradient_plan)) < 1e-7:
            # print('Exiting early, |∂plan| =',abs(jnp.linalg.norm(dplan)))
            break
        # print("A", gradient_steps, abs(jnp.linalg.norm(dplan)))
        #plan -= dplan / jnp.linalg.norm(dplan)
        if isNormalized==1:
            plan -= alpha * gradient_plan / jnp.linalg.norm(gradient_plan)
        else:
            plan -= alpha * gradient_plan

    return plan, gradient_steps

#needs to be fixed
np.random.seed(seed=42)
def get_initial_plans(balloon: Balloon, num_plans, forecast: WindField, atmosphere: JaxAtmosphere, plan_steps, time_delta, stride):
    # flight_record = [(atmosphere.at_pressure(balloon.state.pressure).height.km, 0)]
    flight_record = {atmosphere.at_pressure(balloon.state.pressure).height.km.item(): 0}

    time_to_top = 0
    max_km_to_explore = 19.1

    up_balloon = balloon
    while time_to_top < plan_steps and atmosphere.at_pressure(up_balloon.state.pressure).height.km < max_km_to_explore:
        wind_vector = forecast.get_forecast(up_balloon.state.x/1000, up_balloon.state.y/1000, up_balloon.state.pressure, up_balloon.state.time_elapsed)
        up_balloon = up_balloon.simulate_step_continuous(wind_vector, atmosphere, 0.99, time_delta, stride)
        time_to_top += 1

        flight_record[atmosphere.at_pressure(up_balloon.state.pressure).height.km.item()] = time_to_top

    time_to_bottom = 0
    min_km_to_explore = 15.4

    down_balloon = balloon
    while time_to_bottom < plan_steps and atmosphere.at_pressure(down_balloon.state.pressure).height.km > min_km_to_explore:
        wind_vector = forecast.get_forecast(down_balloon.state.x/1000, down_balloon.state.y/1000, down_balloon.state.pressure, down_balloon.state.time_elapsed)
        down_balloon = down_balloon.simulate_step_continuous(wind_vector, atmosphere, -0.99, time_delta, stride)
        time_to_bottom += 1

        flight_record[atmosphere.at_pressure(down_balloon.state.pressure).height.km.item()] = time_to_bottom
    
    # sorted (should be)
    # flight_record = flight_record_down[::-1] + flight_record_up

    # Sort the dictionary by keys (altitudes) and split them into two separate lists
    sorted_flight_record = sorted(flight_record.items())

    flight_record_altitudes = [altitude for altitude, _ in sorted_flight_record]
    flight_record_steps = [steps for _, steps in sorted_flight_record]
    
    interpolator = scipy.interpolate.RegularGridInterpolator((flight_record_altitudes, ), flight_record_steps, bounds_error=False, fill_value=None)

    plans = []

    for i in range(num_plans):
        random_height = np.random.uniform(15.4, 19.1)
        going_up = random_height >= atmosphere.at_pressure(balloon.state.pressure).height.km
        steps = int(round(interpolator(np.array([random_height]))[0]))
        # print(steps)

        plan = np.zeros((plan_steps, ))
        plan[:steps] = +0.99 if going_up else -0.99 
        # print(random_height, steps)
        try:
            if steps < plan_steps:
                plan[steps:] += np.random.uniform(-0.3, 0.3, plan_steps - steps)
        except:
            print(atmosphere.at_pressure(balloon.state.pressure).height.km.item(), random_height, steps, plan_steps)

        plans.append(plan)
    
    return inverse_sigmoid(np.array(plans))


@partial(jax.jit, static_argnums=(5,6))
@partial(jax.grad, argnums=0)
def get_dplan(plan, cost_fn, balloon: Balloon, wind_field: WindField):
     # jax.debug.print("{balloon}, {wind_field}, {atmosphere}, {terminal_cost_fn}, {time_delta}, {stride}", balloon=balloon, wind_field=wind_field, atmosphere=atmosphere, terminal_cost_fn=terminal_cost_fn, time_delta=time_delta, stride=stride)
    return cost_fn(plan, balloon, wind_field)

class MPC4Agent:
    """
    A Model Predictive Control (MPC) agent that repeatedly forms an optimized plan (plan[i] = the action to take at timestep i) every __ amount of timesteps 
    to find the optimal path from current state to goal state.

    Tasks:
    ??

    State: [lat, long, alt]
    Action: continous value [-1, 1] (is -1 descend, 0 stay, 1 ascend?)

    Optimizer: Gradient Descent/Random Search (for no gradient implementation)
    """
        
    def __init__(self, balloon_env=None): 
        # copied from tree_search_agent.py
        if balloon_env is not None:    # NOTE: this represents the balloon environment for the root node only.
            self.balloon_env = balloon_env
            self.target_lat = balloon_env.target_lat
            self.target_lon = balloon_env.target_lon
            self.target_alt = balloon_env.target_alt
        else:
            print("No environment provided. Initialization failed.")
            return

        self.wind_field = None # WindField
        self.ble_atmosphere = None 
        self.atmosphere = None # Atmosphere

        # self._get_dplan = jax.jit(jax.grad(jax_plan_cost, argnums=0), static_argnames=("time_delta", "stride"))

        self.plan_time = 2*24*60*60
        self.time_delta = 3*60
        self.stride = 10

        # self.plan_steps = 960 + 23 
        self.plan_steps = 240 # (self.plan_time // self.time_delta) // 3
        # self.N = self.plan_steps

        self.plan = None # jnp.full((self.plan_steps, ), fill_value=1.0/3.0)
        self.i = 0

        self.key = jax.random.key(seed=0)
        
        self.balloon = None
        self.time = None
        self.steps_within_radius = 0

    def plan_cost(self, plan, balloon_env: BalloonERAEnvironment, wind_field: WindField):
        """Returns the cost of a plan. The cost is calculated as the euclidean distance between the final state after executing the plan and the target state."""
        #For implementing fly-as-far, could have it be the negative distance from the original starting point
        cost = 0.0

        plan = sigmoid(plan)

        curr_balloon_env = copy.deepcopy(balloon_env)
        final_state = np.array([0.0,0.0,0.0])
        for i in range(len(plan)):
            action = plan[i]
            final_state,reward, done, reason = curr_balloon_env.step(action)
            if done:
                print(f"In cost, reason for stopping: {reason}")
                break
        #jax.debug.print("final state x = {x}, y = {y}, pressure = {p}", x=final_balloon.state.x, y=final_balloon.state.y, p=final_balloon.state.pressure)
        final_state = np.array([curr_balloon_env.lat, curr_balloon_env.lon, curr_balloon_env.alt])
        target_state = np.array([self.target_lat, self.target_lon, self.target_alt])
        cost = np.linalg.norm(final_state- target_state)
        return cost

    def _deadreckon(self):
        """ calculates the current position of a moving object by using a previously determined position, or fix, and incorporating estimates of speed, heading, and elapsed time.
        an observation is a state. it has balloon_obersvation and wind information. we use ob_t and MPC predicts a_t which then gives us ob_t+1
        """
        # wind_vector = self.ble_forecast.get_forecast(
        #     units.Distance(meters=self.balloon.state.x),
        #     units.Distance(meters=self.balloon.state.y), 
        #     self.balloon.state.pressure,
        #     dt.datetime())
        
        # wind_vector = wind_vector.u.meters_per_second, wind_vector.v.meters_per_second
        
        wind_vector = self.wind_field.get_forecast(
            self.balloon.state.x/1000, 
            self.balloon.state.y/1000, 
            self.balloon.state.pressure, 
            self.balloon.state.time_elapsed)
    
        # print(self.balloon.state.time_elapsed/3600.0)

        # print(self.balloon.state.time_elapsed)
        self.balloon = self.balloon.simulate_step_continuous(
            wind_vector, 
            self.atmosphere, 
            self.plan[self.i], 
            self.time_delta, 
            self.stride)
        
        if (self.balloon.state.x/1000)**2 + (self.balloon.state.y/1000)**2 <= (50.0)**2:
            self.steps_within_radius += 1

    def begin_episode(self, observation: np.ndarray) -> int:
        # TODO: actually convert observation into an ndarray (it is a JaxBalloonState, see features.py)
        # balloon = JaxBalloon(jax_balloon_state_from_observation(observation))

        observation: JaxBalloonState = observation
        # if self.balloon is not None:
        #     observation.x = self.balloon.state.x
        #     observation.y = self.balloon.state.y
        #
        self.balloon = Balloon(observation[0],observation[1],observation[2])

        # current_plan_cost = jax_plan_cost(self.plan, balloon, self.forecast, self.atmosphere, self.time_delta, self.stride)
        #if current_plan_cost < best_random_cost:
        #    initial_plan = self.plan

        # TODO: is it necessary to pass in forecast when just trying to get to a height?
        
        initialization_type = 'best_altitude'
        #print('USING ' + initialization_type + ' INITIALIZATION')

        if initialization_type == 'opd':
            start = opd.ExplorerState(
                self.balloon.state.x,
                self.balloon.state.y,
                self.balloon.state.pressure,
                self.balloon.state.time_elapsed)

            search_delta_time = 60*60
            best_node, best_node_early = opd.run_opd_search(start, self.wind_field, [0, 1, 2], opd.ExplorerOptions(budget=25_000, planning_horizon=240, delta_time=search_delta_time))
            initial_plan =  opd.get_plan_from_opd_node(best_node, search_delta_time=search_delta_time, plan_delta_time=self.time_delta)

        elif initialization_type == 'best_altitude':
            if self.plan==None:
                print(f"First")
                self.avg_opt_time=0
                self.avg_iters =0

            initial_plans = get_initial_plans(self.balloon, 100, self.wind_field, self.atmosphere, self.plan_steps, self.time_delta, self.stride)
            
            batched_cost = []
            for i in range(len(initial_plans)):

                batched_cost.append(self.plan_cost(initial_plans[i], self.balloon_env, self.wind_field ))
            
            min_index_so_far = np.argmin(batched_cost)
            min_value_so_far = batched_cost[min_index_so_far]

            initial_plan = initial_plans[min_index_so_far]
            if self.plan is not None and self.plan_cost(self.plan, self.balloon_env, self.wind_field) < min_value_so_far:
                #print('Using the previous optimized plan as initial plan')
                initial_plan = self.plan

            coast = inverse_sigmoid(np.random.uniform(-0.2, 0.2, size=(self.plan_steps, )))
            if self.plan_cost(coast, self.balloon_env, self.wind_field) < min_value_so_far:
                #print('Using the nothing plan as initial plan')
                initial_plan = coast

        elif initialization_type == 'random':
            initial_plan = np.random.uniform(-1.0, 1.0, size=(self.plan_steps, ))
        else:
            initial_plan = np.zeros((self.plan_steps, ))

        optimizing_on = True
        if optimizing_on:
            b4 = time.time()

    
            
            start_cost = self.plan_cost(initial_plan,self.balloon_env, 
                    self.wind_field)
            plan,step= grad_descent_optimizer(initial_plan,get_dplan, 
                    self.balloon, 
                    self.wind_field, 
                    self.atmosphere,
                    self.terminal_cost_fn,
                    self.time_delta, 
                    self.stride)

            after_cost = self.plan_cost(plan, self.balloon, self.wind_field)
            print(f"Gradient descent optimizer, {step}, ∆cost = {after_cost} - {start_cost} = {after_cost - start_cost}")
            print(plan[:15])
            self.plan = plan
            
            
            #print(time.time() - b4, 's to get optimized plan')
            self.plan = sigmoid(self.plan)
            after = time.time()
            print(after - b4, 's to get optimized plan')
        else:
            self.plan = sigmoid(initial_plan)

        self.i = 0

        b4 = time.time()
        self._deadreckon()
        # print(time.time() - b4, 's to deadreckon ballooon')

        action = self.plan[self.i]
        # print('action', action)
        return action.item()

    def step(self, reward: float, observation: np.ndarray) -> int:
        REPLANNING = True
        observation: JaxBalloonState = observation
        self.i+=1
        # self._deadreckon()
        # print(observation.battery_charge/observation.battery_capacity)
        if not REPLANNING:
            self._deadreckon()
            action = self.plan[self.i]
            return action.item()
        else:
            
            N = min(len(self.plan), 23)
            if self.i>0 and self.i%N==0:
                # self.plan_steps -= N
                self.key, rng = jax.random.split(self.key, 2)
                # self.plan = self.plan[N:]
                self.plan = inverse_sigmoid(jnp.hstack((self.plan[N:], jax.random.uniform(rng, (N, ), minval=-0.3, maxval=0.3))))
                #print(self.plan.shape)
                return self.begin_episode(observation)
            else:
                # print('not replanning')
                self._deadreckon()
                action = self.plan[self.i]
                # print('action', action)
                return action.item()
 
    def end_episode(self, reward: float, terminal: bool = True) -> None:
        self.i = 0
        self.steps_within_radius = 0
        self.balloon = None
        self.plan = None
        self.avg_opt_time = 0
        self.avg_iters = 0
        # self.plan_steps = 960 + 23

    #FIX THESE ALSO SELF.FORECAST IS NOW SELF.WIND_FIELD
    def update_forecast(self, forecast: agent.WindField): 
        self.ble_forecast = forecast
        self.forecast = forecast.to_jax_wind_field()

    def update_atmosphere(self, atmosphere: agent.standard_atmosphere.Atmosphere): 
        self.ble_atmosphere = atmosphere
        self.atmosphere = atmosphere.to_jax_atmosphere() 

