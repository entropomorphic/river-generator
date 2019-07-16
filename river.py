import itertools
import functools
from collections import defaultdict
import operator
import time
import hashlib

import numpy as np
import click
import tcod
import tcod.event

TICKS_PER_FRAME = 100

FONT = 'cp437_8x8.png'


@click.command()
@click.option('-w', '--width', default=180, show_default=True)
@click.option('-h', '--height', default=100, show_default=True)
@click.option('-s', '--snakiness', default=2.5, show_default=True, help='Lower values produce tree-like rivers, higher produces long snaky rivers')
@click.option('-t', '--threshold', default=25, show_default=True, help='watershed size below which the river will be hidden')
@click.option('-a', '--animate', is_flag=True, show_default=True, help='Save a screenshot at every frame')
@click.option('--seed', help='seed value for the random generator')
def main(**kwargs):
    '''
    Generate a procedural river network.  Escape to quit, space to restart the generation, [P] to save a screenshot
    '''
    app = RiverApp(**kwargs)
    app.run()


class BitmaskType:
    def __init__(self, **d):
        self._map = d

    def __getattr__(self, name):
        return functools.reduce(operator.or_, [self._map.get(c, 0) for c in name])


Dirs = BitmaskType(N=0b0001, E=0b0010, S=0b0100, W=0b1000)


dir_reverse = defaultdict(lambda: 0, {
    Dirs.N: Dirs.S,
    Dirs.S: Dirs.N,
    Dirs.E: Dirs.W,
    Dirs.W: Dirs.E,
})


dir_offset = defaultdict(lambda: 0, {
    Dirs.N: (0, -1),
    Dirs.S: (0, 1),
    Dirs.E: (1, 0),
    Dirs.W: (-1, 0)
})


def offset(coord, direction):
    ox, oy = dir_offset[direction]
    return coord[0] + ox, coord[1] + oy


# mapping of directions to CP437 box drawing characters
pipe_map = defaultdict(lambda: 250, {
    0: 32,
    Dirs.N: 179,
    Dirs.E: 196,
    Dirs.S: 179,
    Dirs.W: 196,
    Dirs.NE: 192,
    Dirs.NS: 179,
    Dirs.NW: 217,
    Dirs.SE: 218,
    Dirs.EW: 196,
    Dirs.SW: 191,
    Dirs.NES: 195,
    Dirs.NEW: 193,
    Dirs.NSW: 180,
    Dirs.SEW: 194,
    Dirs.NESW: 197
})


# fast sigmoid function using tanh
def sigmoid(x, base=2):
    return 0.5 + np.tanh((x - 0.5) * base * 3.14159) * 0.5


# construct an 8-bit RGBA tuple using possibly [0-1) float values
def color(r, g, b, a=255):
    if isinstance(r, float):
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
    if isinstance(a, float):
        a = int(a * 255)
    return (r, g, b, a)


class RiverGen:
    def __init__(self, seed, width, height, snakiness, threshold):
        self.width, self.height = width, height
        self.snakiness = snakiness
        self.threshold = threshold
        self.parents = np.zeros((width, height), int)
        self.children = np.zeros((width, height), int)
        self.weight = np.zeros((width, height), int)
        self.coords = np.concatenate(np.mgrid[0: self.width, 0: self.height].transpose()).tolist()
        self.random = np.random.RandomState(int(hashlib.md5(str(seed).encode('utf8')).digest()[-4:].hex(), 16))

        # set the initial state. starting point is 2/3 of the way down the left edge
        hh = height // 3
        s1 = (0, height - hh - 1)
        self.parents[s1] = -1
        self.frontier = [s1]

        # mask off the corners of the map to mold the basin into an oblique hexagon shape
        for x in range(hh - 1):
            for y in range((hh - x) * 2):
                self.parents[x, y] = -1
                self.parents[self.width - x - 1, self.height - y - 1] = -1
            for y in range(hh - x):
                self.parents[x, self.height - y - 1] = -1
                self.parents[self.width - x - 1, y] = -1

    def run_tick(self):
        if self.frontier:
            c = len(self.frontier) - 1

            # pick a random frontier cell, weighted to favor the most recently added cells
            # snakiness parameter controls the degree of this weighting
            i = int(c * sigmoid(self.random.random_sample(), self.snakiness))
            fc = self.frontier[i]

            # connect a random neighbor that isn't connected yet, and add it to the frontier.
            for nc, d in self.random.permutation(self.neighbors(fc)):
                if not self.parents[nc]:
                    self.parents[nc] = dir_reverse[d]
                    self.frontier.insert(self.random.randint(i - 1, c), nc)
                    break
            else:
                # if we have no remaining neighbors, remove this cell from the frontier.
                self.frontier.pop(i)

            # recalculate the weights (watershed size) all the way down the river.
            # start drawing juction points when the children's weight is above the threshold.
            p = self.parents[fc]
            while(p > 0):
                nc = offset(fc, p)
                w = self.weight[fc] + 1
                self.weight[fc] = w
                if w >= self.threshold:
                    self.children[nc] |= dir_reverse[p]
                fc = nc
                p = self.parents[fc]

        else:
            return True

    def draw(self, con):
        base_color = color(0.66, 0.66, 1)
        for x, y in self.coords:
            coord = (x, y)
            weight = self.weight[coord]
            c = base_color

            if weight < self.threshold:
                continue

            # for drawn tiles, the intensity is the cube root of the weight.
            # It just looks nice that way.
            elif weight < 4096:
                w = (weight ** 0.33 / 16)
                c = color(w * 0.66, w * 0.66, w)

            con.tiles[y, x] = (
                pipe_map[self.parents[coord] | self.children[coord]],
                c,
                (0, 0, 0, 255)
            )

    def neighbors(self, coord):
        x, y = coord
        ns = []
        if x > 0:
            ns.append(((x - 1, y), Dirs.W))
        if x < self.width - 1:
            ns.append(((x + 1, y), Dirs.E))
        if y > 0:
            ns.append(((x, y - 1), Dirs.N))
        if y < self.height - 1:
            ns.append(((x, y + 1), Dirs.S))
        return ns


class RiverApp:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.gen = None
        self.running = True
        self.animate = kwargs.pop('animate', False)
        self.screenshot_number = 1

        tcod.console_set_custom_font(FONT, tcod.FONT_TYPE_GREYSCALE | tcod.FONT_LAYOUT_ASCII_INROW)
        self.root = tcod.console_init_root(kwargs['width'], kwargs['height'], 'RIVER TEST', False)

        self.restart()

    def run(self):
        tcod.console_flush()
        for tick in itertools.count():
            if self.gen:
                done = self.gen.run_tick()
                if not tick % TICKS_PER_FRAME:
                    self.gen.draw(self.root)
                    tcod.console_flush()
                    if self.animate:
                        self.save_screenshot()

                    self.handle_input()
                if done:
                    ct = time.time() - self.start_time
                    print(f'Completed in {ct:0.2f} seconds')
                    self.gen = None
            else:
                self.handle_input(True)

            if not self.running:
                break

    def handle_input(self, block=False):
        getter = tcod.event.wait if block else tcod.event.get
        for event in getter():
            if event.type == 'QUIT':
                self.running = False
            elif event.type == 'KEYUP':
                if event.sym == tcod.event.K_ESCAPE:
                    self.running = False
                elif event.sym == ord(' '):
                    self.restart()
                elif event.sym in (ord('P'), ord('p')):
                    n = self.save_screenshot()
                    print(f'Screenshot saved to {n}')

    def restart(self):
        self.root.clear()
        self.start_time = time.time()

        self.seed = self.options.pop('seed', None)
        if self.seed is None:
            self.seed = np.random.bytes(4).hex()

        print(f'Starting; seed {str(self.seed)}')
        self.gen = RiverGen(seed=self.seed, **self.options)

    def save_screenshot(self):
        n = f'River-{self.seed}-{self.screenshot_number:04}.png'
        self.screenshot_number += 1
        tcod.sys_save_screenshot(n)
        return n


if __name__ == '__main__':
    main()
