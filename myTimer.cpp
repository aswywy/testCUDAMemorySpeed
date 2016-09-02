#include "myTimer.h"

myTimer::myTimer()
{
	sdkCreateTimer(&_timer);
}

myTimer::~myTimer() 
{
	sdkDeleteTimer(&_timer);
}

void myTimer::start() 
{
	sdkStartTimer(&_timer);
}

void myTimer::stop() 
{
	sdkStopTimer(&_timer);
}

double myTimer::duration_in_seconds() const 
{
	return _timer->getTime() / 1000.0;
}

double myTimer::duration_in_milliseconds() const {
	return _timer->getTime();
}

ScopedTimer::ScopedTimer(const char* str)
: mStr(str) {
	mTimer.start();
}

ScopedTimer::~ScopedTimer() {
	mTimer.stop();
	if (mTimer.duration_in_milliseconds() > 0)
	{
		printf("%s: %f ms\n", mStr.c_str(), mTimer.duration_in_milliseconds());
	}

}