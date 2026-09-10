# python code

from qpanda3_runtime import RuntimeService
service = RuntimeService()
service.login('f97ec9d9f061b6388f018ff498bfd3c3bd9ea9e67584115a6742005bb48f71c8393446526451534c6c61676876707554')

# 使用RuntimeService.list_devices查询访问通道支持的计算后端设备
devs = service.list_devices()
print('devs:',devs)
