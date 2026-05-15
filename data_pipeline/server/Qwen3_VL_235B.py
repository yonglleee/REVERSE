from proto.astra import astrastarlinkbroker_pb2
from proto.astra import astracomm_pb2
from proto.astra import astraagent_pb2
from proto import mmbridge_pb2

import logging
import py3meshkit

# 必须放在py3meshkit的后面
from comm2.tlvpickle import skbuiltintype_pb2

import random
import socket
import struct
import asyncio
import base64
from openai import OpenAI
import glob

import time


starlink_broker_route_parser = py3meshkit.RouteParser("astrastarlinkbroker")
mmbridge_route_parser = py3meshkit.RouteParser("mmbridge", "shanghai")


async def schedule_fast_agent_job(
    app_id: str, agent_req: astraagent_pb2.ProcessReq, timeout_ms: int
):
  uin = random.randint(10000, 65536)
  get_machine_req = astrastarlinkbroker_pb2.GetMachineForJobReq()
  output_buf = b""

  job = get_machine_req.job
  job.appid = app_id

  client = py3meshkit.SvrkitClient("astrastarlinkbroker", route_parser=starlink_broker_route_parser)
  get_machine_resp = astrastarlinkbroker_pb2.GetMachineForJobResp()
  print('get_machine_req',get_machine_req)
  ret = await client.request(uin, 1, get_machine_req, get_machine_resp)
  if ret != 0:
      logging.error("GetMachineForJob fail, ret = {}".format(ret))
      return ret, output_buf
  machine = get_machine_resp.machine
  if not machine.machine_id:
      logging.error("ret = {}, machine_id = {}".format(ret, machine.machine_id))
      return 1, output_buf



  ip = socket.inet_ntoa(struct.pack("!I", machine.ip))
  # port = machine.port
  # print('machine',ip)
  # print('machine',port)
  return ip
  
# 上面的代码都不要动！！
# 下面是实际调用的例子，可以参考着来, 有问题咨询 rickrong


class Qwen3VL235B:
  def __init__(self, service="GLM_4_5", ip=None) -> None:
    self.service = service
    self.ip = ip
  
  def __call__(self, content, ip=None):
    if ip is None:
      print("Qwen3VL235B getting ip...")
      ip = asyncio.run(schedule_fast_agent_job(self.service, 'test', 60000)) # 设置app_id，具体可以咨询 rickrong
    print(f"Qwen3VL235B: {ip=}")
    # ip = self.ip
    
    openai_api_key = "EMPTY"
    openai_api_base = "http://{}:8080/v1".format(ip)
    client = OpenAI(
      api_key=openai_api_key,
      base_url=openai_api_base,
    )
    return self.req_agent(client, content)

  def req_agent(self, client, content, ip=None):
    """
    content:[
                  {
                      "type": "image_url",
                      "image_url": {
                          "url": base64_qwen
                      },
                  },
                  {"type": "text", "text": prompt},
              ],
    """
    model_name = client.models.list().data[0].id
    chat_response = client.chat.completions.create(
      model=model_name,
      messages=[
        {
          "role": "user",
          "content": content,
        },
      ],
      temperature=0.000001,
      top_p=1.0,
    )
    # print("Chat response:", chat_response.choices[0].message.content)
    return chat_response.choices[0].message.content
      



