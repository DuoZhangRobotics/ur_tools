#!/bin/sh
# Source ROS 2 Jazzy and the dedicated ur_tools uv environment from sh or zsh.

if [ ! -f /opt/ros/jazzy/setup.sh ]; then
  echo "ROS 2 Jazzy is not installed at /opt/ros/jazzy" >&2
  return 1 2>/dev/null || exit 1
fi

UR_TOOLS_ROOT=${UR_TOOLS_ROOT:-/home/duo/ur_tools}
if [ ! -f "$UR_TOOLS_ROOT/.venv/bin/activate" ]; then
  echo "ur_tools uv environment is missing at $UR_TOOLS_ROOT/.venv" >&2
  return 1 2>/dev/null || exit 1
fi

# setup.bash cannot determine its own location when sourced by zsh. The POSIX
# setup uses this prefix when no stale value is inherited.
unset AMENT_CURRENT_PREFIX
. /opt/ros/jazzy/setup.sh
. "$UR_TOOLS_ROOT/.venv/bin/activate"
unset UR_TOOLS_ROOT
