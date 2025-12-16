#!/bin/bash
echo "Listing candidate processes:"
ps aux | egrep 'python|ros|fastdds|cyclonedds|rtps|eprosima|unitree|mujoco' | grep -v grep

read -p "Kill listed PIDs? (y/N) " ans
if [[ "$ans" == "y" ]]; then
  ps aux | egrep 'python|ros|fastdds|cyclonedds|rtps|eprosima|unitree|mujoco' | grep -v grep | awk '{print $2}' | xargs -r sudo kill -9
fi

echo "Listing /dev/shm entries with common DDS names:"
ls -lah /dev/shm | egrep 'rtps|fastrtps|eprosima|cyclone|dds|fastdds' || true
read -p "Remove these /dev/shm entries? (y/N) " ans2
if [[ "$ans2" == "y" ]]; then
  ls -lah /dev/shm | egrep 'rtps|fastrtps|eprosima|cyclone|dds|fastdds' | awk '{print $9}' | xargs -r sudo rm -f
fi