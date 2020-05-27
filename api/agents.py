from spade import agent
from spade.behaviour import CyclicBehaviour
import asyncio


class ChatbotAgent(agent.Agent):
    class FindRecommendations(CyclicBehaviour):
        async def on_start(self):
            print("Starting behaviour . . .")
            self.counter = 0

        async def run(self):
            print("Counter: {}".format(self.counter))
            self.counter += 1
            await asyncio.sleep(1)

    async def setup(self):
        print("Agent starting . . .")
        b = self.FindRecommendations()
        self.add_behaviour(b)
