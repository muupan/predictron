import argparse

import chainer
from chainer import optimizers
import numpy as np

from predictron import Predictron


def generate_maze(size=20):
    # 1 = wall, 0 = empty
    # A maze will contain 30% of walls
    maze = (np.random.rand(size, size) < 0.3).astype(np.float32)
    maze[0, 0] = 0
    maze[-1, -1] = 0
    return maze


def check_connectivity_to_right_bottom(maze):
    maze = np.pad(maze, 1, 'constant', constant_values=1)  # Pad with walls
    visited = np.zeros_like(maze)
    dst = (maze.shape[0] - 2, maze.shape[1] - 2)

    def search(pos):
        if visited[pos]:
            return
        if maze[pos]:
            return
        visited[pos] = 1
        search((pos[0] + 1, pos[1]))
        search((pos[0] - 1, pos[1]))
        search((pos[0], pos[1] + 1))
        search((pos[0], pos[1] - 1))

    search(dst)
    return np.diagonal(visited[1:-1, 1:-1])


def phi(maze):
    # 3 binary channels: wall, empty, field (full of ones except padding)
    return np.asarray([maze, -maze, np.ones_like(maze)])


def generate_supervised_batch(maze_size=20, batch_size=100):
    xs = []
    ts = []
    for b in range(batch_size):
        maze = generate_maze(maze_size)
        connectivity = check_connectivity_to_right_bottom(maze)
        xs.append(phi(maze))
        ts.append(connectivity)
    x = np.asarray(xs, dtype=np.float32)
    t = np.asarray(ts, dtype=np.float32)
    return x, t


def generate_unsupervised_batch(maze_size=20, batch_size=100):
    xs = []
    for b in range(batch_size):
        maze = generate_maze(maze_size)
        xs.append(phi(maze))
    x = np.asarray(xs, dtype=np.float32)
    return x


def main():

    parser = argparse.ArgumentParser(description='Predictron on random mazes')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of transitions in each mini-batch')
    parser.add_argument('--max-iter', type=int, default=10000,
                        help='Number of iterations to run')
    parser.add_argument('--n-model-steps', type=int, default=16,
                        help='Number of model steps')
    parser.add_argument('--n-channels', type=int, default=32,
                        help='Number of channels for hidden units')
    parser.add_argument('--maze-size', type=int, default=20,
                        help='Size of random mazes')
    parser.add_argument('--use-reward-gamma', type=bool, default=True,
                        help='Use reward and gamma')
    parser.add_argument('--use-lambda', type=bool, default=True,
                        help='Use lambda-network')
    parser.add_argument('--usage-weighting', type=bool, default=True,
                        help='Enable usage weighting')
    parser.add_argument('--n-unsupervised-updates', type=int, default=0,
                        help='Number of unsupervised upates per supervised'
                             'updates')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()

    # chainer.set_debug(True)
    model = Predictron(n_tasks=args.maze_size, n_channels=args.n_channels,
                       model_steps=args.n_model_steps,
                       use_reward_gamma=args.use_reward_gamma,
                       use_lambda=args.use_lambda,
                       usage_weighting=args.usage_weighting)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)
    opt = optimizers.Adam()
    opt.setup(model)

    for i in range(args.max_iter):
        x, t = generate_supervised_batch(
            maze_size=args.maze_size, batch_size=args.batchsize)
        if args.gpu >= 0:
            x = chainer.cuda.to_gpu(x)
            t = chainer.cuda.to_gpu(t)
        model.cleargrads()
        g_k_loss, g_lambda_loss = model.supervised_loss(x, t)
        supervised_loss = g_k_loss + g_lambda_loss
        supervised_loss.backward()
        opt.update()
        for _ in range(args.n_unsupervised_updates):
            x = generate_unsupervised_batch(
                maze_size=args.maze_size, batch_size=args.batchsize)
            if args.gpu >= 0:
                x = chainer.cuda.to_gpu(x)
            model.cleargrads()
            unsupervised_loss = model.unsupervised_loss(x)
            unsupervised_loss.backward()
            opt.update()
        print(i, g_k_loss.data, g_lambda_loss.data,
              (g_lambda_loss.data ** 0.5))


if __name__ == '__main__':
    main()
