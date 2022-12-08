class EpisodeLengthTimer:
    def __init__(self, max_episode_length=1000):
        self._max_episode_length, self._curr_episode_length = max_episode_length, 0

    def reset(self) -> None:
        self._curr_episode_length = 0

    def advance(self) -> bool:
        self._tick()
        return self._has_elapsed()

    def _tick(self) -> None:
        self._curr_episode_length += 1

    def _has_elapsed(self) -> bool:
        if self._curr_episode_length > self._max_episode_length:
            return True
        return False
