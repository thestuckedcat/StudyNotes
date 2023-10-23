#ifndef MYSTRING_H_
#define MYSTRING_H_
#include<string>
// overload by member method
class Mystring {
	friend std::ostream& operator<<(std::ostream& os, const Mystring& rhs);
	friend std::istream& operator>>(std::istream& in, Mystring& rhs);
private:
	char* str;
public:
	Mystring();							//No-args constructor
	Mystring(const char* s);			//Overloaded constructor
	Mystring(const Mystring& source);	//Copy constructor
	Mystring(Mystring&& source);		//Move constructor
	~Mystring();						//Destructor

	Mystring& operator=(const Mystring& rhs);	//Copy assignment
	Mystring& operator=(Mystring&& rhs);		//Move assignment

	Mystring operator-()const; // return lowercase version of object's string
	bool operator==(const Mystring &rhs)const; // return if two strings are not equal
	bool operator!=(const Mystring &rhs)const;
	bool operator<(const Mystring &rhs)const;// if lhs lexically less than rhs
	bool operator>(const Mystring &rhs)const;// if lhs lexically greater than rhs
	Mystring operator+(const Mystring& rhs)const;// concatenates
	Mystring& operator+=(const Mystring& rhs);//concatenate
	Mystring operator*(const int& repeat_num)const;
	Mystring& operator*=(const int& repeat_num);
	Mystring& operator++();
	Mystring operator++(int);

	void display() const;

	int get_length() const;                                                // getters
	const char* get_str() const;

};




#endif

