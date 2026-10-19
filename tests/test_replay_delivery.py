"""A browser with a stalled socket must not stall the GPU replay producer."""
import asyncio
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server import Demo


class SlowClient:
    closed = False

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.messages = []

    async def send_str(self, value):
        self.entered.set()
        await self.release.wait()
        self.messages.append(json.loads(value))


class ReplayDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_stalled_viewer_does_not_block_producer_and_receives_latest_frame(self):
        demo = Demo('http://127.0.0.1:9000', ROOT/'data/nuscenes-long')
        client = SlowClient()
        pending, event = {}, asyncio.Event()
        demo.clients[client] = pending, event
        sender = asyncio.create_task(demo.send_updates(client, pending, event))
        try:
            await demo.broadcast({'type': 'frame', 'index': 1})
            await asyncio.wait_for(client.entered.wait(), 1)
            await asyncio.wait_for(demo.broadcast({'type': 'frame', 'index': 2}), 1)
            await asyncio.wait_for(demo.broadcast({'type': 'frame', 'index': 3}), 1)
            await asyncio.wait_for(demo.broadcast({'type': 'status', 'playing': True}), 1)
            self.assertEqual(len(pending), 2)
            client.release.set()
            for _ in range(10):
                if len(client.messages) == 3:
                    break
                await asyncio.sleep(0)
            self.assertEqual([m['index'] for m in client.messages if m['type'] == 'frame'], [1, 3])
            self.assertEqual(client.messages[-1], {'type': 'status', 'playing': True})
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)

    async def test_pc_replay_starts_paused_and_local_replay_starts_playing(self):
        pc = Demo('http://127.0.0.1:9000', ROOT/'data/nuscenes-long')
        local = Demo('http://127.0.0.1:9000', ROOT/'data/nuscenes-long', local=True)
        self.assertFalse(pc.state['playing'])
        self.assertTrue(local.state['playing'])
