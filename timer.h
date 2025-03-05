/* Developed by Jimmy Hu */

#ifndef Timer_H
#define Timer_H

#include <chrono>
#include <iostream>
//#include <print>

namespace TinyDIP
{
    class Timer
    {
    private:
        std::chrono::system_clock::time_point start, end;
        std::chrono::duration<double> elapsed_seconds;
        std::time_t end_time;
    public:
        Timer()
        {
            start = std::chrono::system_clock::now();
        }

        ~Timer()
        {
            end = std::chrono::system_clock::now();
            elapsed_seconds = end - start;
            end_time = std::chrono::system_clock::to_time_t(end);
            if (elapsed_seconds.count() != 1)
            {
                //std::print(std::cout, "Computation finished at {} elapsed time: {} seconds.\n", std::ctime(&end_time), elapsed_seconds.count());
                std::cout << "Computation finished at " << std::ctime(&end_time) << " elapsed time: " << elapsed_seconds.count() << "seconds.\n";
            }
            else
            {
                //std::print(std::cout, "Computation finished at {} elapsed time: {} second.\n", std::ctime(&end_time), elapsed_seconds.count());
                std::cout << "Computation finished at " << std::ctime(&end_time) << " elapsed time: " << elapsed_seconds.count() << "second.\n";
            }
        }

    };
}

#endif