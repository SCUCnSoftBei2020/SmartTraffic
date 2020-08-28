//
// Created by Zilin Xiao on 2020/6/5.
//

#ifndef DARKNET_EXCEPTION_H
#define DARKNET_EXCEPTION_H
#include <exception>
#include <cstdio>

class FileNotFound : public std::exception{

public:
    explicit FileNotFound(const char* c){
        m_p = const_cast<char *>(c);
    }
    virtual char* what(){
        printf("FileNotFound: %s", m_p);
        return m_p;
    }
private:
    char* m_p;
};

#endif //DARKNET_EXCEPTION_H
