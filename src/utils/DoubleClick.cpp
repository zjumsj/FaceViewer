#include "DoubleClick.h"

#include <time.h>
#include <sys/timeb.h>

//sizeof(long long int)=8

glutDoubleClick::glutDoubleClick(){
	_interval = 0.1f;
	_lasttime = 0ll;
}

glutDoubleClick::glutDoubleClick(float interval){
	_interval = interval;
	_lasttime = 0ll;
}

long long int glutDoubleClick::getCurrentTime(){
	struct timeb timer_msec;
	long long int timestamp_msec;
	ftime(&timer_msec);
	timestamp_msec=((long long int) timer_msec.time * 1000ll ) + (long long int)timer_msec.millitm;
	return timestamp_msec;
}

glutDoubleClick::MouseOperation::MouseOperation(){
	button = -1;
	state = -1;
	x = -1;
	y = -1;
}

bool glutDoubleClick::setMouseOperation(MouseOperation nowop, MouseOperation & lastop){
	lastop = _lastop;
	_lastop = nowop;
	long long int nowtime = getCurrentTime();
	bool result = (nowtime - _lasttime) < static_cast<long long int>(_interval * 1000);
	_lasttime = nowtime;
	return result;
}

//////////////////////////////////
glutTimeInterval::glutTimeInterval(){
	_lasttime = getCurrentTime();
}

int glutTimeInterval::touch(bool clear){
	long long int currenttime =  getCurrentTime();
	int interval=static_cast<int>(currenttime - _lasttime);
	if(clear)
		_lasttime = currenttime;
	return interval;
}

long long int glutTimeInterval::getCurrentTime()
{
	struct timeb timer_msec;
	long long int timestamp_msec;
	ftime(&timer_msec);
	timestamp_msec=((long long int) timer_msec.time * 1000ll ) + (long long int)timer_msec.millitm;
	return timestamp_msec;
}

bool glutTimeInterval::compareThreshold(int ms, bool clear) {
	long long int currenttime = getCurrentTime();
	int interval = static_cast<int>(currenttime - _lasttime);
	bool state = false;
	if (interval >= ms){
		state = true;
		if (clear)
			_lasttime = currenttime;		
	}
	return state;
}
