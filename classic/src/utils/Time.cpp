// Modified by Gravity Lab contributors, 2026-08-28: match the on-disk case so
// the source builds on case-sensitive filesystems and without Apple warnings.
#include "Time.h"

#include <SDL2/SDL.h>
#include <chrono>

namespace Time {
int64_t currentTimeMillis()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
}

void sleep(int64_t ms)
{
    SDL_Delay(ms);
}
}
