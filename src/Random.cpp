/*
*/
#include <Random.hpp>
using namespace zeuron;
/*
 */
std::random_device Random::_randomDevice;
std::mt19937 Random::_mt19937(Random::_randomDevice());
/*
 */