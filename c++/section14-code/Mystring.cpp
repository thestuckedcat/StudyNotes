#define _CRT_SECURE_NO_WARNINGS
#include"Mystring.h"
#include<iostream>


using namespace std;



std::ostream& operator<<(std::ostream& os, const Mystring& rhs)
{
	os << rhs.str << endl;
	return os;
}

std::istream& operator>>(std::istream& in, Mystring& rhs)
{
	char* buff = new char[1000];
	in >> buff;
	rhs = Mystring{ buff };
	delete[] buff;
	return in;
}


Mystring::Mystring()	//No-args constructor
	:str{nullptr}		//好习惯，防止野指针
{
	str = new char[1];
	*str = '\0';
}
Mystring::Mystring(const char* s)//Overloaded constructor
	:str{nullptr}
{
	if (s == nullptr) {
		str = new char[1];
		*str = '\0';
	}
	else {
		str = new char[strlen(s) + 1];
		strcpy(str, s);
	}
}

Mystring::Mystring(const Mystring& source)	//Copy constructor
{
	str = new char[strlen(source.str) + 1];
	strcpy(str, source.str);
}

Mystring::Mystring(Mystring&& source)		//Move constructor
	:str{source.str}
{
	source.str = nullptr;
}

Mystring::~Mystring()						//Destructor
{
	delete [] str;
}

Mystring& Mystring::operator=(const Mystring& rhs)	//Copy assignment
{
	if (this == &rhs) {
		return *this;
	}
	else {
		delete[] str;
		str = new char[strlen(rhs.str) + 1];
		strcpy(str, rhs.str);
		return *this;
	}
}

Mystring& Mystring::operator=(Mystring&& rhs)		//Move assignment
{
	if (this == &rhs) {
		return *this;
	}
	delete [] str;
	str = rhs.str;
	rhs.str = nullptr;
	return *this;
}

Mystring Mystring::operator-() const // return lowercase version of object's string
{
	//提供一个处理好的副本，不改变原序列：snew = -s_old
	char* buff = new char[strlen(str) + 1];
	strcpy(buff, str);
	for (size_t i = 0; i < strlen(buff); i++) {
		buff[i] = tolower(buff[i]);
	}
	Mystring temp{ buff };
	delete [] buff;
	return temp;
}

bool Mystring::operator==(const Mystring& rhs) const // return if two strings are not equal
{
	return (strcmp(str, rhs.str)==0);
}

bool Mystring::operator!=(const Mystring& rhs) const
{
	return !(*this == rhs);
}

bool Mystring::operator<(const Mystring& rhs) const// if lhs lexically less than rhs
{
	return (strcmp(str, rhs.str) < 0);
}

bool Mystring::operator>(const Mystring& rhs) const// if lhs lexically greater than rhs
{
	return (strcmp(str, rhs.str) > 0);
}

Mystring Mystring::operator+(const Mystring& rhs) const// concatenates
{
	char* buff = new char[strlen(str) + strlen(rhs.str) + 1];
	strcpy(buff, str);
	strcat(buff, rhs.str);
	Mystring temp{ buff };
	delete[] buff;
	return temp;
}

Mystring& Mystring::operator+=(const Mystring& rhs)//concatenate
{
	*this = *this + rhs;
	return *this;
}

Mystring Mystring::operator*(const int& repeat_num)const
{
	char* buff = new char[strlen(str) * repeat_num + 1];
	strcpy(buff, str);
	for (int i = 1; i < repeat_num; i++) {
		strcat(buff, str);
	}
	/*
	Mystring temp;
    for (int i=1; i<= n; i++)
        temp = temp + *this;
	*/
	return buff;
}

Mystring& Mystring::operator*=(const int& repeat_num)
{
	*this = *this * repeat_num;
	return *this;
}

Mystring& Mystring::operator++()//pre
{
	for (int i = 0; i < strlen(str); i++) {
		str[i] = toupper(str[i]);
	}
	return *this;
}

Mystring Mystring::operator++(int)//post-increment,返回原对象副本，然后将对象++
{
	Mystring temp{ *this };
	operator++();
	return temp;
}

// Display method
void Mystring::display() const {
	std::cout << str << " : " << get_length() << std::endl;
}

// getters
int Mystring::get_length() const { return strlen(str); }
const char* Mystring::get_str() const { return str; }