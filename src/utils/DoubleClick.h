#pragma once

class glutDoubleClick{
public:
	struct MouseOperation{
		int button;
		int state;
		int x;
		int y;
		MouseOperation();
	};

	glutDoubleClick();
	glutDoubleClick(float interval);

	//true if within the time
	//false otherwise
	bool setMouseOperation(MouseOperation nowop,MouseOperation & lastop);

protected:
	long long int getCurrentTime();

private:
	float _interval;
	MouseOperation _lastop;
	long long int _lasttime;

};

class glutTimeInterval{
public:
	glutTimeInterval();
	int touch(bool clear = true);
	bool compareThreshold(int ms,bool clear = true);
protected:
	long long int getCurrentTime();
private:
	long long int _lasttime;
};