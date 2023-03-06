#include "mtest.h"


int main()
{
    testprint();
    testsum(2,3);

    Teststruct * s;
    s = malloc(sizeof(Teststruct));
    s = init_teststruct(s);
    print_teststruct(s);

    return 0;
}