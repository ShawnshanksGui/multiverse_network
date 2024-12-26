#pragma once

#include "sim.hpp"

namespace madEscape {

// Creates agents, outer walls and floor. Entities that will persist across
// all episodes.
void createPersistentEntities(Engine &ctx);

// Randomly generate a new world for a training episode
// First, destroys any non-persistent state for the current world and then
// generates a new play area.
void generateWorld(Engine &ctx);




//
using madrona::Entity;
using madrona::CountT; 

void generate_switch(Engine &ctx, CountT k_ary);

void generate_in_port(Engine &ctx, CountT k_ary);

void generate_e_port(Engine &ctx, CountT k_ary);

void generate_host(Engine &ctx, CountT k_ary);

//
}
