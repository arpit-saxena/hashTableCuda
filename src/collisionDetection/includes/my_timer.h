#ifndef MY_TIMER_H
#define MY_TIMER_H

#include <chrono>
namespace sc = std::chrono;

class MyTimer {
	int numSessions = 0;
	sc::steady_clock::duration totalTimeElapsed;
	sc::steady_clock::duration currentTimeElapsed;
	sc::steady_clock::time_point startTime;
	bool stopped = true;

public:
	MyTimer(): totalTimeElapsed(0), currentTimeElapsed(0) {}
	void start();
	void stop();
	void reset();
	float getTime(); // current time in milli seconds
	float getAverageTime();
};

void MyTimer::start() {
	stopped = false;
	startTime = sc::steady_clock::now();
	numSessions++;
}

void MyTimer::stop() {
	if (stopped) return;
	auto stopTime = sc::steady_clock::now();
	currentTimeElapsed = stopTime - startTime;
	totalTimeElapsed += currentTimeElapsed;
	stopped = true;
}

void MyTimer::reset() {
	stop();
	numSessions = 0;
	totalTimeElapsed = sc::steady_clock::duration{0};
	currentTimeElapsed = sc::steady_clock::duration{0};
}

float MyTimer::getTime() {
	auto time = sc::steady_clock::now();
	auto ret = stopped ? currentTimeElapsed : time - startTime;
	return sc::duration_cast<sc::milliseconds>(ret).count();
}

float MyTimer::getAverageTime() {
	auto ret = totalTimeElapsed;
	if (!stopped) ret += sc::steady_clock::now() - startTime;
	if (numSessions == 0) return 0;
	printf("%d %d\n", numSessions, sc::duration_cast<sc::milliseconds>(ret).count() / numSessions);
	return sc::duration_cast<sc::milliseconds>(ret).count() / numSessions;
}

#endif /* MY_TIMER_H */
