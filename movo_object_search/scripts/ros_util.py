# Copyright 2022 Kaiyu Zheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rospy

def get_param(param):
    
    if rospy.has_param(param):
        return rospy.get_param(param)
    else:
        return rospy.get_param("~" + param)


def get_if_has_param(param):
    
    if rospy.has_param(param):
        return rospy.get_param(param)
    elif rospy.has_param("~" + param):
        return rospy.get_param("~" + param)
    else:
        return None

