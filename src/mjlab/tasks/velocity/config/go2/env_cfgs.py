""" unitree Go2 velocity env cofig."""

import math

from typing import Literal

from mjlab.asset_zoo.robots import(

GO2_ACTION_SCALE,
get_go2_robot_cfg,

)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TerrainType = Literal["rough", "obstacles"]

def unitree_go2_rough_env_cfg(
    play: bool= False,
) -> ManagerBasedRlEnvCfg: 
    
    cfg =make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.mujoco.impratio = 10
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.contact_sensor_maxmatch= 500

    cfg.scene.entities ={"robot": get_go2_robot_cfg()}
    
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan" :
           assert isinstance(sensor, RayCastSensorCfg)
           assert isinstance(sensor.frame, ObjRef)
           sensor.frame.name = "base_link"
    
    foot_names = ("FR", "FL", "RR", "RL")
    site_names = ("FR", "FL", "RR", "RL")
    geom_names = tuple(f"{name}_foot_collision" for name in foot_names)


    for sensor in cfg.scene.sensors or ():
         if sensor.name == "foot_height_scan":
             assert isinstance(sensor, TerrainHeightSensorCfg)
             sensor.frame = tuple(
             ObjRef(type="site", name=s, entity="robot") for s in site_names
             )
             sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)
    
    feet_ground_cfg = ContactSensorCfg(

         name="feet_ground_contact",
         primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
         secondary=ContactMatch(mode="body", pattern="terrain"),
         fields=("found", "force"),
         reduce="netforce",
         num_slots=1,
         track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(

         name="self_collision",
         primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
         secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
         fields=("found", "force"),
         reduce="none",
         num_slots=1,
         history_length=4,
    )

    """ self collision cfg"""


    calf_geom_names = tuple(
         f"{leg}_calf{i}_collision" for leg in foot_names for i in (1, 2)
    )

    shank_ground_cfg = ContactSensorCfg(

         name="shank_ground_touch",
         primary=ContactMatch(
             mode="geom",
             entity="robot",
             pattern=calf_geom_names,
         ),
         secondary=ContactMatch(mode="body", pattern="terrain"),
         fields=("found", "force"),
         reduce="none",
         num_slots=1,
         history_length=4,
    )  

    """ calf collision register name and auto match .*CALF GEMO MESH name in go2 xml asset """


    base_link_ground_cfg = ContactSensorCfg(

        name="base_link_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=("base1_collision","base2_collision","base3_collision"),            
        ),

         secondary=ContactMatch(mode="body", pattern="terrain"),
         fields=("found", "force"),
         reduce="none",
         num_slots=1,
         history_length=4,    

    )

    """ base_link 0 1 2 collision register name aligned with go2 xml asset"""


    thigh_geom_names = tuple(
         f"{leg}_thigh_collision" for leg in foot_names
    )

    thigh_ground_cfg = ContactSensorCfg(
         name="thigh_ground_touch",
         primary=ContactMatch(
             mode="geom",
             entity="robot",
             pattern=thigh_geom_names,
         ),
         secondary=ContactMatch(mode="body", pattern="terrain"),
         fields=("found", "force"),
         reduce="none",
         num_slots=1,
         history_length=4,
    )


    """ thigh collision register name and auto match .*THIGH GEMO MESH name in go2 xml asset """



    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
       feet_ground_cfg,
       self_collision_cfg,
       thigh_ground_cfg,
       shank_ground_cfg,
       base_link_ground_cfg,
     )
    
    """ add sensor to all of collision"""

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = GO2_ACTION_SCALE
 
    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0

    # Replace the base foot_friction with per-axis friction events for condim 6.
    del cfg.events["foot_friction"]
    cfg.events["foot_friction_slide"] = EventTermCfg(
           mode="startup",
           func=envs_mdp.dr.geom_friction,
           params={
                    "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
                    "operation": "abs",
                    "axes": [0],
                    "ranges": (0.3, 1.5),
                    "shared_random": True,
               },
           )
    cfg.events["foot_friction_spin"] = EventTermCfg(
           mode="startup",
           func=envs_mdp.dr.geom_friction,
           params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
                "operation": "abs",
                "distribution": "log_uniform",
                "axes": [1],
                "ranges": (1e-4, 2e-2),
                "shared_random": True,
          },
    )
    cfg.events["foot_friction_roll"] = EventTermCfg(
          mode="startup",
          func=envs_mdp.dr.geom_friction,
          params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
                "operation": "abs",
                "distribution": "log_uniform",
                "axes": [2],
                "ranges": (1e-5, 5e-3),
                "shared_random": True,
          },
    )

    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

    cfg.rewards["pose"].params["std_standing"] = {
           r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
           r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
    }
    cfg.rewards["pose"].params["std_walking"] = {
           r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
           r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
    }
    cfg.rewards["pose"].params["std_running"] = {
           r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
           r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
    }

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
    cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

    for reward_name in ["foot_clearance", "foot_slip"]:
           cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    cfg.rewards["body_ang_vel"].weight = 0.0
    cfg.rewards["angular_momentum"].weight = 0.0
    cfg.rewards["air_time"].weight = 0.0

    # Per-body-group collision penalties.
    cfg.rewards["self_collisions"] = RewardTermCfg(
           func=mdp.self_collision_cost,
           weight=-0.1,
           params={"sensor_name": self_collision_cfg.name},
    )

    cfg.rewards["shank_collision"] = RewardTermCfg(
           func=mdp.self_collision_cost,
           weight=-0.1,
           params={"sensor_name": shank_ground_cfg.name},
    )


        
    cfg.rewards["base_link_ground_collision"] = RewardTermCfg(
           func=mdp.self_collision_cost,
           weight=-0.1,
           params={"sensor_name": base_link_ground_cfg.name},
    )

    # On rough terrain the quadruped tilts significantly; don't terminate on
    # orientation alone. Let out_of_terrain_bounds handle resets.
    cfg.terminations.pop("fell_over", None)

    cfg.terminations["illegal_contact"] = TerminationTermCfg(
           func=mdp.illegal_contact,
           params={"sensor_name": thigh_ground_cfg.name},
    )

      # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
          cfg.episode_length_s = int(1e9)

          cfg.observations["actor"].enable_corruption = False
          cfg.events.pop("push_robot", None)
          cfg.terminations.pop("out_of_terrain_bounds", None)
          cfg.curriculum = {}
          cfg.events["randomize_terrain"] = EventTermCfg(
                func=envs_mdp.randomize_terrain,
                mode="reset",
                params={},
          )

          if cfg.scene.terrain is not None:
                if cfg.scene.terrain.terrain_generator is not None:
                     cfg.scene.terrain.terrain_generator.curriculum = False
                     cfg.scene.terrain.terrain_generator.num_cols = 5
                     cfg.scene.terrain.terrain_generator.num_rows = 5
                     cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg
