// MoveIt checks the actual URDF geometry; OMPL searches for a collision-free path.
// No controllers, publishers, or Gazebo state are accessed by this module.
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <geometric_shapes/shapes.h>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>

namespace py = pybind11;
namespace ob = ompl::base;
namespace og = ompl::geometric;
using Q = std::array<double, 4>;
using Box = std::array<double, 6>;  // centre XYZ, size XYZ, all in base_link metres
using Centre = std::array<double, 3>;
using Path = std::vector<Q>;
constexpr double PI = 3.14159265358979323846;
constexpr double CHECK_STEP = 0.002;  // radians; also check between output waypoints
const std::array<std::string, 4> JOINTS = {
  "base_rotation_joint", "shoulder_joint", "elbow_joint", "wrist_pitch_joint"};

class Planner
{
public:
  Planner(const std::string& urdf_xml, const std::string& srdf_xml)
  {
    auto urdf = urdf::parseURDF(urdf_xml);
    auto srdf = std::make_shared<srdf::Model>();
    if (!urdf || !srdf->initString(*urdf, srdf_xml))
      throw std::invalid_argument("Cannot load the arm URDF/SRDF");
    model_ = std::make_shared<moveit::core::RobotModel>(urdf, srdf);
    for (const auto& name : JOINTS)
      if (!model_->hasJointModel(name)) throw std::invalid_argument("Missing arm joint: " + name);
  }

  void validate(const Path& path, double finger, const std::vector<Box>& boxes,
                const std::vector<Centre>& cubes, double cube_size,
                std::optional<double> payload_offset) const
  {
    auto scene = make_scene(boxes, cubes, cube_size, payload_offset);
    moveit::core::RobotState state(model_);
    state.setToDefaultValues();
    if (path.empty()) throw std::invalid_argument("The path is empty");
    if (!valid(*scene, state, path.front(), finger))
      throw std::invalid_argument("Path starts in collision or outside joint limits: " + contact(*scene, state));
    for (std::size_t i = 1; i < path.size(); ++i)
      if (!motion_valid(*scene, state, path[i-1], path[i], finger))
        throw std::invalid_argument("Path would collide or exceed joint limits: " + contact(*scene, state));
  }

  Path plan(const Q& start, const Q& goal, double finger, const std::vector<Box>& boxes,
            const std::vector<Centre>& cubes, double cube_size,
            std::optional<double> payload_offset, bool keep_down) const
  {
    auto scene = make_scene(boxes, cubes, cube_size, payload_offset);
    moveit::core::RobotState robot(model_);
    robot.setToDefaultValues();
    if (!valid(*scene, robot, start, finger))
      throw std::invalid_argument("Start pose is invalid: " + contact(*scene, robot));
    if (!valid(*scene, robot, goal, finger))
      throw std::invalid_argument("Goal pose is invalid: " + contact(*scene, robot));
    Q first = start;
    if (keep_down)
    {
      // On this arm, downward orientation means shoulder + elbow + wrist = pi.
      // Search three independent joints and compute the fourth exactly, rather
      // than letting an unconstrained planner tilt a carried cube.
      first[3] = PI-first[1]-first[2];
      if (std::abs(first[3]-start[3]) > 0.04 ||
          std::abs(goal[1]+goal[2]+goal[3]-PI) > 1e-6)
        throw std::invalid_argument("A carrying path needs a downward start and goal");
      if (!motion_valid(*scene, robot, start, first, finger))
        throw std::invalid_argument("Cannot safely level the gripper at the start");
    }
    Path result{start};
    if (motion_valid(*scene, robot, first, goal, finger))
    {
      result.push_back(first); result.push_back(goal);
      return densify(result);
    }
    const unsigned int dimensions = keep_down ? 3 : 4;
    auto space = std::make_shared<ob::RealVectorStateSpace>(dimensions);
    ob::RealVectorBounds bounds(dimensions);
    for (unsigned int i = 0; i < dimensions; ++i)
    {
      const auto& limit = model_->getVariableBounds(JOINTS[i]);
      bounds.setLow(i, limit.min_position_); bounds.setHigh(i, limit.max_position_);
    }
    space->setBounds(bounds);
    og::SimpleSetup setup(space);
    auto read = [keep_down, dimensions](const ob::State* state) {
      Q q{};
      const auto* values = state->as<ob::RealVectorStateSpace::StateType>();
      for (unsigned int i=0; i<dimensions; ++i) q[i]=values->values[i];
      if (keep_down) q[3]=PI-q[1]-q[2];
      return q;
    };
    scene->getCollisionEnvNonConst()->setPadding(0.003);  // Extra planning buffer before final validation.
    setup.setStateValidityChecker([&](const ob::State* state) {
      return valid(*scene, robot, read(state), finger);
    });
    setup.getSpaceInformation()->setStateValidityCheckingResolution(
        CHECK_STEP/(2.0*space->getMaximumExtent()));
    ob::ScopedState<> from(space), to(space);
    for (unsigned int i=0; i<dimensions; ++i) { from[i]=first[i]; to[i]=goal[i]; }
    setup.setStartAndGoalStates(from, to, 1e-8);
    auto search = std::make_shared<og::RRTConnect>(setup.getSpaceInformation());
    search->setRange(0.2);
    setup.setPlanner(search);
    if (setup.solve(3.0) != ob::PlannerStatus::EXACT_SOLUTION)
      throw std::invalid_argument("No exact collision-free route found within 3 seconds");
    setup.simplifySolution(0.1);
    for (const auto* state : setup.getSolutionPath().getStates()) result.push_back(read(state));
    result.push_back(goal);
    result = densify(result);
    // Check the exact piecewise-linear path the Python executor will follow.
    validate(result, finger, boxes, cubes, cube_size, payload_offset);
    return result;
  }

private:
  moveit::core::RobotModelPtr model_;

  planning_scene::PlanningScenePtr make_scene(const std::vector<Box>& boxes,
      const std::vector<Centre>& cubes, double cube_size, std::optional<double> offset) const
  {
    if (!std::isfinite(cube_size) || cube_size <= 0)
      throw std::invalid_argument("Cube size must be positive and finite");
    auto scene = std::make_shared<planning_scene::PlanningScene>(model_);
    scene->getCollisionEnvNonConst()->setPadding(0.002);
    auto add = [&](const std::string& name, const shapes::ShapeConstPtr& shape, const Centre& centre) {
      for (double v : centre) if (!std::isfinite(v)) throw std::invalid_argument("Invalid obstacle centre");
      Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
      pose.translation() = Eigen::Vector3d(centre[0], centre[1], centre[2]);
      scene->getWorldNonConst()->addToObject(name, shape, pose);
    };
    add("floor", std::make_shared<shapes::Box>(4.,4.,.1), {0.,0.,-.05});
    scene->getAllowedCollisionMatrixNonConst().setEntry("base_link", "floor", true);
    for (std::size_t i=0; i<boxes.size(); ++i)
    {
      const auto& b=boxes[i];
      for (unsigned int j=3; j<6; ++j)
        if (!std::isfinite(b[j]) || b[j] <= 0) throw std::invalid_argument("Invalid obstacle size");
      add("box_"+std::to_string(i), std::make_shared<shapes::Box>(b[3],b[4],b[5]), {b[0],b[1],b[2]});
    }
    // The enclosing cylinder is conservative for a cube with unknown yaw.
    for (std::size_t i=0; i<cubes.size(); ++i)
      add("cube_"+std::to_string(i), std::make_shared<shapes::Cylinder>(cube_size/std::sqrt(2.),cube_size), cubes[i]);
    if (offset)
    {
      if (!std::isfinite(*offset) || *offset <= 0)
        throw std::invalid_argument("Invalid carried-cube offset");
      moveit_msgs::msg::AttachedCollisionObject attached;
      attached.link_name = "tool0";
      attached.touch_links = {"left_finger_link", "right_finger_link"};
      attached.object.header.frame_id = "tool0";
      attached.object.id = "carried_cube";
      attached.object.operation = moveit_msgs::msg::CollisionObject::ADD;
      shape_msgs::msg::SolidPrimitive shape;
      shape.type = shape.CYLINDER;
      shape.dimensions = {cube_size, cube_size/std::sqrt(2.)};
      geometry_msgs::msg::Pose pose;
      pose.position.z = *offset; pose.orientation.w = 1.;
      attached.object.primitives.push_back(shape); attached.object.primitive_poses.push_back(pose);
      if (!scene->processAttachedCollisionObjectMsg(attached))
        throw std::invalid_argument("Cannot add the carried cube to the planning scene");
    }
    return scene;
  }

  bool valid(const planning_scene::PlanningScene& scene, moveit::core::RobotState& state,
             const Q& q, double finger) const
  {
    if (!std::isfinite(finger) || finger<-.000001 || finger>.035001) return false;
    for (unsigned int i=0; i<4; ++i)
    {
      if (!std::isfinite(q[i])) return false;
      state.setVariablePosition(JOINTS[i],q[i]);
    }
    state.setVariablePosition("gripper_opening_joint",std::clamp(finger,0.,.035));
    state.update();
    if (!state.satisfiesBounds()) return false;
    collision_detection::CollisionRequest request;
    collision_detection::CollisionResult result;
    scene.checkCollision(request, result, state);
    return !result.collision;
  }

  bool motion_valid(const planning_scene::PlanningScene& scene, moveit::core::RobotState& state,
                    const Q& a, const Q& b, double finger) const
  {
    double change=0.;
    for (unsigned int j=0;j<4;++j)
    {
      const auto& limit=model_->getVariableBounds(JOINTS[j]);
      if (!std::isfinite(b[j]) || b[j]<limit.min_position_ || b[j]>limit.max_position_) return false;
      change=std::max(change,std::abs(b[j]-a[j]));
    }
    unsigned int steps=std::max(1u,static_cast<unsigned int>(std::ceil(change/CHECK_STEP)));
    for (unsigned int i=0;i<=steps;++i)
    {
      Q q{};
      for (unsigned int j=0;j<4;++j) q[j]=a[j]+(b[j]-a[j])*i/steps;
      if (!valid(scene,state,q,finger)) return false;
    }
    return true;
  }

  static std::string contact(const planning_scene::PlanningScene& scene, moveit::core::RobotState& state)
  {
    collision_detection::CollisionRequest request;
    request.contacts=true; request.max_contacts=1;
    collision_detection::CollisionResult result;
    scene.checkCollision(request,result,state);
    if (result.contacts.empty()) return "joint bounds or invalid input";
    const auto& pair=result.contacts.begin()->first;
    return pair.first+" / "+pair.second;
  }

  static Path densify(const Path& points)
  {
    Path result{points.front()};
    for (std::size_t i=1;i<points.size();++i)
    {
      double change=0.;
      for (unsigned int j=0;j<4;++j) change=std::max(change,std::abs(points[i][j]-points[i-1][j]));
      const unsigned int steps=std::max(1u,static_cast<unsigned int>(std::ceil(change/.01)));
      for (unsigned int k=1;k<=steps;++k)
      {
        Q q{};
        for (unsigned int j=0;j<4;++j) q[j]=points[i-1][j]+(points[i][j]-points[i-1][j])*k/steps;
        result.push_back(q);
      }
    }
    return result;
  }
};

PYBIND11_MODULE(_planner, module)
{
  // MoveIt's logger needs a C++ ROS context even inside a rclpy process.
  // Leave signal handling to rclpy and clean up only a context we created.
  const bool owned = !rclcpp::ok();
  if (owned) rclcpp::init(0, nullptr, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  module.add_object("_context", py::capsule(new bool(owned), [](void* p) {
    std::unique_ptr<bool> owner(static_cast<bool*>(p));
    if (*owner && rclcpp::ok()) rclcpp::shutdown();
  }));
  py::class_<Planner>(module,"Planner")
    .def(py::init<const std::string&,const std::string&>())
    .def("plan",&Planner::plan,py::call_guard<py::gil_scoped_release>())
    .def("validate",&Planner::validate,py::call_guard<py::gil_scoped_release>());
}
