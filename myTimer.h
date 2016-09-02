#include <string>
#include <helper_functions.h>

#pragma once
class myTimer
{
public:
	myTimer();
	~myTimer();

	/**
	* Start the timer.
	*/
	void start();

	/**
	* Stop the timer.
	*/
	void stop();

	/**
	* Get the timer in seconds.
	*/
	double duration_in_seconds() const;

	/**
	* Get the timer in milliseconds.
	*/
	double duration_in_milliseconds() const;

private:
	StopWatchInterface *_timer = NULL;
}; // Timer

/**
* Implements a scoped timer.
*/
class ScopedTimer {
public:

	ScopedTimer(const char* str);
	~ScopedTimer();

	operator bool() {
		return true;
	}

private:

	// The actual timer
	myTimer mTimer;
	// Description
	std::string mStr;

}; // ScopedTimer
#define TIMED(X) if(ScopedTimer _ScopedTimer = X)
