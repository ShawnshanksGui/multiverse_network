#include "level_gen.hpp"

//
#include "sim.hpp"
#include "2_fib_nexthop_init_mflow.hpp"
#include "2_flow_mflow.hpp"
//

namespace madEscape {

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace consts {

inline constexpr float doorWidth = consts::worldWidth / 3.f;

}

enum class RoomType : uint32_t {
    SingleButton,
    DoubleButton,
    CubeBlocking,
    CubeButtons,
    NumTypes,
};

static inline float randInRangeCentered(Engine &ctx, float range)
{
    return ctx.data().rng.sampleUniform() * range - range / 2.f;
}

static inline float randBetween(Engine &ctx, float min, float max)
{
    return ctx.data().rng.sampleUniform() * (max - min) + min;
}

// Initialize the basic components needed for physics rigid body entities
static inline void setupRigidBodyEntity(
    Engine &ctx,
    Entity e,
    Vector3 pos,
    Quat rot,
    SimObject sim_obj,
    EntityType entity_type,
    ResponseType response_type = ResponseType::Dynamic,
    Diag3x3 scale = {1, 1, 1})
{
    ObjectID obj_id { (int32_t)sim_obj };

    ctx.get<Position>(e) = pos;
    ctx.get<Rotation>(e) = rot;
    ctx.get<Scale>(e) = scale;
    ctx.get<ObjectID>(e) = obj_id;
    ctx.get<Velocity>(e) = {
        Vector3::zero(),
        Vector3::zero(),
    };
    ctx.get<ResponseType>(e) = response_type;
    ctx.get<ExternalForce>(e) = Vector3::zero();
    ctx.get<ExternalTorque>(e) = Vector3::zero();
    ctx.get<EntityType>(e) = entity_type;
}

// Register the entity with the broadphase system
// This is needed for every entity with all the physics components.
// Not registering an entity will cause a crash because the broadphase
// systems will still execute over entities with the physics components.
static void registerRigidBodyEntity(
    Engine &ctx,
    Entity e,
    SimObject sim_obj)
{
    ObjectID obj_id { (int32_t)sim_obj };
    ctx.get<broadphase::LeafID>(e) =
        PhysicsSystem::registerEntity(ctx, e, obj_id);
}

// Creates floor, outer walls, and agent entities.
// All these entities persist across all episodes.
void createPersistentEntities(Engine &ctx)
{
    // Create the floor entity, just a simple static plane.
    ctx.data().floorPlane = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().floorPlane,
        Vector3 { 0, 0, 0 },
        Quat { 1, 0, 0, 0 },
        SimObject::Plane,
        EntityType::None, // Floor plane type should never be queried
        ResponseType::Static);

    // Create the outer wall entities
    // Behind
    ctx.data().borders[0] = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().borders[0],
        Vector3 {
            0,
            -consts::wallWidth / 2.f,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::worldWidth + consts::wallWidth * 2,
            consts::wallWidth,
            2.f,
        });

    // Right
    ctx.data().borders[1] = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().borders[1],
        Vector3 {
            consts::worldWidth / 2.f + consts::wallWidth / 2.f,
            consts::worldLength / 2.f,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::wallWidth,
            consts::worldLength,
            2.f,
        });

    // Left
    ctx.data().borders[2] = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().borders[2],
        Vector3 {
            -consts::worldWidth / 2.f - consts::wallWidth / 2.f,
            consts::worldLength / 2.f,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::wallWidth,
            consts::worldLength,
            2.f,
        });

    // Create agent entities. Note that this leaves a lot of components
    // uninitialized, these will be set during world generation, which is
    // called for every episode.
    // for (CountT i = 0; i < consts::numAgents; ++i) {
    for (CountT i = 0; i < 1; ++i) {
        Entity agent = ctx.data().agents[i] =
            ctx.makeRenderableEntity<Agent>();

        // Create a render view for the agent
        if (ctx.data().enableRender) {
            render::RenderingSystem::attachEntityToView(ctx,
                    agent,
                    100.f, 0.001f,
                    1.5f * math::up);
        }

        ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };
        ctx.get<ObjectID>(agent) = ObjectID { (int32_t)SimObject::Agent };
        ctx.get<ResponseType>(agent) = ResponseType::Dynamic;
        ctx.get<GrabState>(agent).constraintEntity = Entity::none();
        ctx.get<EntityType>(agent) = EntityType::Agent;

        ctx.get<CurStep>(agent).step = 0;

    }

    // Populate OtherAgents component, which maintains a reference to the
    // other agents in the world for each agent.
    // for (CountT i = 0; i < consts::numAgents; i++) {
    for (CountT i = 0; i < 1; ++i) {
        Entity cur_agent = ctx.data().agents[i];

        OtherAgents &other_agents = ctx.get<OtherAgents>(cur_agent);
        CountT out_idx = 0;
        for (CountT j = 0; j < consts::numAgents; j++) {
            if (i == j) {
                continue;
            }

            Entity other_agent = ctx.data().agents[j];
            other_agents.e[out_idx++] = other_agent;
        }
    }




}

// Although agents and walls persist between episodes, we still need to
// re-register them with the broadphase system and, in the case of the agents,
// reset their positions.
static void resetPersistentEntities(Engine &ctx)
{
    registerRigidBodyEntity(ctx, ctx.data().floorPlane, SimObject::Plane);

     for (CountT i = 0; i < 3; i++) {
         Entity wall_entity = ctx.data().borders[i];
         registerRigidBodyEntity(ctx, wall_entity, SimObject::Wall);
     }

     for (CountT i = 0; i < consts::numAgents; i++) {
         Entity agent_entity = ctx.data().agents[i];
         registerRigidBodyEntity(ctx, agent_entity, SimObject::Agent);

         // Place the agents near the starting wall
         Vector3 pos {
             randInRangeCentered(ctx, 
                 consts::worldWidth / 2.f - 2.5f * consts::agentRadius),
             randBetween(ctx, consts::agentRadius * 1.1f,  2.f),
             0.f,
         };

         if (i % 2 == 0) {
             pos.x += consts::worldWidth / 4.f;
         } else {
             pos.x -= consts::worldWidth / 4.f;
         }

         ctx.get<Position>(agent_entity) = pos;
         ctx.get<Rotation>(agent_entity) = Quat::angleAxis(
             randInRangeCentered(ctx, math::pi / 4.f),
             math::up);

         auto &grab_state = ctx.get<GrabState>(agent_entity);
         if (grab_state.constraintEntity != Entity::none()) {
             ctx.destroyEntity(grab_state.constraintEntity);
             grab_state.constraintEntity = Entity::none();
         }

         ctx.get<Progress>(agent_entity).maxY = pos.y;

         ctx.get<Velocity>(agent_entity) = {
             Vector3::zero(),
             Vector3::zero(),
         };
         ctx.get<ExternalForce>(agent_entity) = Vector3::zero();
         ctx.get<ExternalTorque>(agent_entity) = Vector3::zero();
         ctx.get<Action>(agent_entity) = Action {
             .moveAmount = 0,
             .moveAngle = 0,
             .rotate = consts::numTurnBuckets / 2,
             .grab = 0,
         };

         ctx.get<StepsRemaining>(agent_entity).t = consts::episodeLen;
     }
}

// Builds the two walls & door that block the end of the challenge room
static void makeEndWall(Engine &ctx,
                        Room &room,
                        CountT room_idx)
{
    float y_pos = consts::roomLength * (room_idx + 1) -
        consts::wallWidth / 2.f;

    // Quarter door of buffer on both sides, place door and then build walls
    // up to the door gap on both sides
    float door_center = randBetween(ctx, 0.75f * consts::doorWidth, 
        consts::worldWidth - 0.75f * consts::doorWidth);
    float left_len = door_center - 0.5f * consts::doorWidth;
    Entity left_wall = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        left_wall,
        Vector3 {
            (-consts::worldWidth + left_len) / 2.f,
            y_pos,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            left_len,
            consts::wallWidth,
            1.75f,
        });
    registerRigidBodyEntity(ctx, left_wall, SimObject::Wall);

    float right_len =
        consts::worldWidth - door_center - 0.5f * consts::doorWidth;
    Entity right_wall = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        right_wall,
        Vector3 {
            (consts::worldWidth - right_len) / 2.f,
            y_pos,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            right_len,
            consts::wallWidth,
            1.75f,
        });
    registerRigidBodyEntity(ctx, right_wall, SimObject::Wall);

    Entity door = ctx.makeRenderableEntity<DoorEntity>();
    setupRigidBodyEntity(
        ctx,
        door,
        Vector3 {
            door_center - consts::worldWidth / 2.f,
            y_pos,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Door,
        EntityType::Door,
        ResponseType::Static,
        Diag3x3 {
            consts::doorWidth * 0.8f,
            consts::wallWidth,
            1.75f,
        });
    registerRigidBodyEntity(ctx, door, SimObject::Door);
    ctx.get<OpenState>(door).isOpen = false;

    room.walls[0] = left_wall;
    room.walls[1] = right_wall;
    room.door = door;
}

static Entity makeButton(Engine &ctx,
                         float button_x,
                         float button_y)
{
    Entity button = ctx.makeRenderableEntity<ButtonEntity>();
    ctx.get<Position>(button) = Vector3 {
        button_x,
        button_y,
        0.f,
    };
    ctx.get<Rotation>(button) = Quat { 1, 0, 0, 0 };
    ctx.get<Scale>(button) = Diag3x3 {
        consts::buttonWidth,
        consts::buttonWidth,
        0.2f,
    };
    ctx.get<ObjectID>(button) = ObjectID { (int32_t)SimObject::Button };
    ctx.get<ButtonState>(button).isPressed = false;
    ctx.get<EntityType>(button) = EntityType::Button;

    return button;
}

static Entity makeCube(Engine &ctx,
                       float cube_x,
                       float cube_y,
                       float scale = 1.f)
{
    Entity cube = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        cube,
        Vector3 {
            cube_x,
            cube_y,
            1.f * scale,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Cube,
        EntityType::Cube,
        ResponseType::Dynamic,
        Diag3x3 {
            scale,
            scale,
            scale,
        });
    registerRigidBodyEntity(ctx, cube, SimObject::Cube);

    return cube;
}

static void setupDoor(Engine &ctx,
                      Entity door,
                      Span<const Entity> buttons,
                      bool is_persistent)
{
    DoorProperties &props = ctx.get<DoorProperties>(door);

    for (CountT i = 0; i < buttons.size(); i++) {
        props.buttons[i] = buttons[i];
    }
    props.numButtons = (int32_t)buttons.size();
    props.isPersistent = is_persistent;
}

// A room with a single button that needs to be pressed, the door stays open.
static CountT makeSingleButtonRoom(Engine &ctx,
                                   Room &room,
                                   float y_min,
                                   float y_max)
{
    float button_x = randInRangeCentered(ctx,
        consts::worldWidth / 2.f - consts::buttonWidth);
    float button_y = randBetween(ctx, y_min + consts::roomLength / 4.f,
        y_max - consts::wallWidth - consts::buttonWidth / 2.f);

    Entity button = makeButton(ctx, button_x, button_y);

    setupDoor(ctx, room.door, { button }, true);

    room.entities[0] = button;

    return 1;
}

// A room with two buttons that need to be pressed simultaneously,
// the door stays open.
static CountT makeDoubleButtonRoom(Engine &ctx,
                                   Room &room,
                                   float y_min,
                                   float y_max)
{
    float a_x = randBetween(ctx,
        -consts::worldWidth / 2.f + consts::buttonWidth,
        -consts::buttonWidth);

    float a_y = randBetween(ctx,
        y_min + consts::roomLength / 4.f,
        y_max - consts::wallWidth - consts::buttonWidth / 2.f);

    Entity a = makeButton(ctx, a_x, a_y);

    float b_x = randBetween(ctx,
        consts::buttonWidth,
        consts::worldWidth / 2.f - consts::buttonWidth);

    float b_y = randBetween(ctx,
        y_min + consts::roomLength / 4.f,
        y_max - consts::wallWidth - consts::buttonWidth / 2.f);

    Entity b = makeButton(ctx, b_x, b_y);

    setupDoor(ctx, room.door, { a, b }, true);

    room.entities[0] = a;
    room.entities[1] = b;

    return 2;
}

// This room has 3 cubes blocking the exit door as well as two buttons.
// The agents either need to pull the middle cube out of the way and
// open the door or open the door with the buttons and push the cubes
// into the next room.
static CountT makeCubeBlockingRoom(Engine &ctx,
                                   Room &room,
                                   float y_min,
                                   float y_max)
{
    float button_a_x = randBetween(ctx,
        -consts::worldWidth / 2.f + consts::buttonWidth,
        -consts::buttonWidth - consts::worldWidth / 4.f);

    float button_a_y = randBetween(ctx,
        y_min + consts::buttonWidth,
        y_max - consts::roomLength / 4.f);

    Entity button_a = makeButton(ctx, button_a_x, button_a_y);

    float button_b_x = randBetween(ctx,
        consts::buttonWidth + consts::worldWidth / 4.f,
        consts::worldWidth / 2.f - consts::buttonWidth);

    float button_b_y = randBetween(ctx,
        y_min + consts::buttonWidth,
        y_max - consts::roomLength / 4.f);

    Entity button_b = makeButton(ctx, button_b_x, button_b_y);

    setupDoor(ctx, room.door, { button_a, button_b }, true);

    Vector3 door_pos = ctx.get<Position>(room.door);

    float cube_a_x = door_pos.x - 3.f;
    float cube_a_y = door_pos.y - 2.f;

    Entity cube_a = makeCube(ctx, cube_a_x, cube_a_y, 1.5f);

    float cube_b_x = door_pos.x;
    float cube_b_y = door_pos.y - 2.f;

    Entity cube_b = makeCube(ctx, cube_b_x, cube_b_y, 1.5f);

    float cube_c_x = door_pos.x + 3.f;
    float cube_c_y = door_pos.y - 2.f;

    Entity cube_c = makeCube(ctx, cube_c_x, cube_c_y, 1.5f);

    room.entities[0] = button_a;
    room.entities[1] = button_b;
    room.entities[2] = cube_a;
    room.entities[3] = cube_b;
    room.entities[4] = cube_c;

    return 5;
}

// This room has 2 buttons and 2 cubes. The buttons need to remain pressed
// for the door to stay open. To progress, the agents must push at least one
// cube onto one of the buttons, or more optimally, both.
static CountT makeCubeButtonsRoom(Engine &ctx,
                                  Room &room,
                                  float y_min,
                                  float y_max)
{
    float button_a_x = randBetween(ctx,
        -consts::worldWidth / 2.f + consts::buttonWidth,
        -consts::buttonWidth - consts::worldWidth / 4.f);

    float button_a_y = randBetween(ctx,
        y_min + consts::buttonWidth,
        y_max - consts::roomLength / 4.f);

    Entity button_a = makeButton(ctx, button_a_x, button_a_y);

    float button_b_x = randBetween(ctx,
        consts::buttonWidth + consts::worldWidth / 4.f,
        consts::worldWidth / 2.f - consts::buttonWidth);

    float button_b_y = randBetween(ctx,
        y_min + consts::buttonWidth,
        y_max - consts::roomLength / 4.f);

    Entity button_b = makeButton(ctx, button_b_x, button_b_y);

    setupDoor(ctx, room.door, { button_a, button_b }, false);

    float cube_a_x = randBetween(ctx,
        -consts::worldWidth / 4.f,
        -1.5f);

    float cube_a_y = randBetween(ctx,
        y_min + 2.f,
        y_max - consts::wallWidth - 2.f);

    Entity cube_a = makeCube(ctx, cube_a_x, cube_a_y, 1.5f);

    float cube_b_x = randBetween(ctx,
        1.5f,
        consts::worldWidth / 4.f);

    float cube_b_y = randBetween(ctx,
        y_min + 2.f,
        y_max - consts::wallWidth - 2.f);

    Entity cube_b = makeCube(ctx, cube_b_x, cube_b_y, 1.5f);

    room.entities[0] = button_a;
    room.entities[1] = button_b;
    room.entities[2] = cube_a;
    room.entities[3] = cube_b;

    return 4;
}

// Make the doors and separator walls at the end of the room
// before delegating to specific code based on room_type.
static void makeRoom(Engine &ctx,
                     LevelState &level,
                     CountT room_idx,
                     RoomType room_type)
{
    Room &room = level.rooms[room_idx];
    makeEndWall(ctx, room, room_idx);

    float room_y_min = room_idx * consts::roomLength;
    float room_y_max = (room_idx + 1) * consts::roomLength;

    CountT num_room_entities;
    switch (room_type) {
    case RoomType::SingleButton: {
        num_room_entities =
            makeSingleButtonRoom(ctx, room, room_y_min, room_y_max);
    } break;
    case RoomType::DoubleButton: {
        num_room_entities =
            makeDoubleButtonRoom(ctx, room, room_y_min, room_y_max);
    } break;
    case RoomType::CubeBlocking: {
        num_room_entities =
            makeCubeBlockingRoom(ctx, room, room_y_min, room_y_max);
    } break;
    case RoomType::CubeButtons: {
        num_room_entities =
            makeCubeButtonsRoom(ctx, room, room_y_min, room_y_max);
    } break;
    default: MADRONA_UNREACHABLE();
    }

    // Need to set any extra entities to type none so random uninitialized data
    // from prior episodes isn't exported to pytorch as agent observations.
    for (CountT i = num_room_entities; i < consts::maxEntitiesPerRoom; i++) {
        room.entities[i] = Entity::none();
    }
}

static void generateLevel(Engine &ctx)
{
    LevelState &level = ctx.singleton<LevelState>();

    // For training simplicity, define a fixed sequence of levels.
    makeRoom(ctx, level, 0, RoomType::DoubleButton);
    makeRoom(ctx, level, 1, RoomType::CubeBlocking);
    makeRoom(ctx, level, 2, RoomType::CubeButtons);

#if 0
    // An alternative implementation could randomly select the type for each
    // room rather than a fixed progression of challenge difficulty
    for (CountT i = 0; i < consts::numRooms; i++) {
        RoomType room_type = (RoomType)(
            ctx.data().rng.sampleI32(0, (uint32_t)RoomType::NumTypes));

        makeRoom(ctx, level, i, room_type);
    }
#endif
}

// Randomly generate a new world for a training episode
void generateWorld(Engine &ctx)
{
    resetPersistentEntities(ctx);
    generateLevel(ctx);
}


















void generate_switch(Engine &ctx, CountT k_ary)
{
    CountT num_edge_switch = k_ary*k_ary/2;
    CountT num_aggr_switch = k_ary*k_ary/2; // the number of aggragate switch
    CountT num_core_switch = k_ary*k_ary/4;

    CountT num_host = k_ary*k_ary*k_ary/4;
    CountT num_next_hop = k_ary;

    printf("*******generate_switch*******\n");
    // build edge switch
    for (uint32_t i = 0; i < num_edge_switch; i++) {
        Entity e_switch = ctx.makeEntity<Switch>();
        printf("");
        ctx.get<SwitchType>(e_switch) = SwitchType::Edge;
        ctx.get<SwitchID>(e_switch).switch_id = i;

        for (uint32_t j = 0; j < num_host; j++) {
            for (uint32_t k = 0; k < num_next_hop; k++) {
                ctx.get<FIBTable>(e_switch).fib_table[j][k] = central_fib_table[i][j][k];
            }
        }
        
        ctx.get<QueueNumPerPort>(e_switch).queue_num_per_port = QUEUE_NUM;

        ctx.data()._switches[ctx.data().numSwitch++] = e_switch;
    }

    // build aggragate switch
    for (uint32_t i = 0; i < num_aggr_switch; i++) {
        Entity e_switch = ctx.makeEntity<Switch>();
        ctx.get<SwitchType>(e_switch) = SwitchType::Aggr;
        ctx.get<SwitchID>(e_switch).switch_id = i+num_edge_switch;

        for (uint32_t j = 0; j < num_host; j++) {
            for (uint32_t k = 0; k < num_next_hop; k++) {
                ctx.get<FIBTable>(e_switch).fib_table[j][k] = central_fib_table[i+num_edge_switch][j][k];
            }
        }

        ctx.get<QueueNumPerPort>(e_switch).queue_num_per_port = QUEUE_NUM;

        ctx.data()._switches[ctx.data().numSwitch++] = e_switch;
    }

    // build core switch
    for (uint32_t i = 0; i < num_core_switch; i++) {
        Entity e_switch = ctx.makeEntity<Switch>();
        ctx.get<SwitchType>(e_switch) = SwitchType::Core;
        ctx.get<SwitchID>(e_switch).switch_id = i + num_edge_switch + num_aggr_switch;

        for (uint32_t j = 0; j < num_host; j++) {
            for (uint32_t k = 0; k < num_next_hop; k++) {
                ctx.get<FIBTable>(e_switch).fib_table[j][k] = central_fib_table[i+num_edge_switch+num_aggr_switch][j][k];
            }
        }

        ctx.get<QueueNumPerPort>(e_switch).queue_num_per_port = QUEUE_NUM;
        
        ctx.data()._switches[ctx.data().numSwitch++] = e_switch;
    }
    //printf("\n");
    printf("Total %d switches, %d in_ports, %d e_ports\n", \
           ctx.data().numSwitch, ctx.data().numInPort, ctx.data().numEPort);
}


void generate_in_port(Engine &ctx, CountT k_ary)
{
    CountT num_edge_switch = k_ary*k_ary/2;
    CountT num_aggr_switch = k_ary*k_ary/2; // the number of aggragate switch
    CountT num_core_switch = k_ary*k_ary/4;

    printf("*******Generate in ports*******\n");
    // ingress_ports for edge switch
    ctx.data().numInPort = 0;
    for (uint32_t i = 0; i < num_edge_switch; i++) {
        for (uint32_t j = 0; j < k_ary; j++) {
            Entity in_port = ctx.makeEntity<IngressPort>();
            ctx.get<PortType>(in_port) = PortType::InPort;
            ctx.get<LocalPortID>(in_port).local_port_id = j;
            ctx.get<GlobalPortID>(in_port).global_port_id = i*k_ary+j;

            ctx.get<PktBuf>(in_port).head = 0;
            ctx.get<PktBuf>(in_port).tail = 0;
            ctx.get<PktBuf>(in_port).cur_num = 0;
            ctx.get<PktBuf>(in_port).cur_bytes = 0;

            ctx.get<SwitchID>(in_port).switch_id = i;
            
            ctx.get<SimTime>(in_port).sim_time = 0;
            ctx.get<SimTimePerUpdate>(in_port).sim_time_per_update = LOOKAHEAD_TIME; //1000ns

            ctx.data().inPorts[ctx.data().numInPort++] = in_port;
        }
    }

    // ingress_ports for aggragate switch
    for (uint32_t i = 0; i < num_aggr_switch; i++) {
        for (uint32_t j = 0; j < k_ary; j++) {
            Entity in_port = ctx.makeEntity<IngressPort>();
            ctx.get<PortType>(in_port) = PortType::InPort;
            ctx.get<LocalPortID>(in_port).local_port_id = j;
            ctx.get<GlobalPortID>(in_port).global_port_id = (num_edge_switch+i)*k_ary+j;

            ctx.get<PktBuf>(in_port).head = 0;
            ctx.get<PktBuf>(in_port).tail = 0;
            ctx.get<PktBuf>(in_port).cur_num = 0;
            ctx.get<PktBuf>(in_port).cur_bytes = 0;

            ctx.get<SwitchID>(in_port).switch_id = i + num_edge_switch;

            // printf("switch_id: %d, local_port_id: %d, global_port_id: %d\n", \
            //         ctx.get<SwitchID>(in_port).switch_id, ctx.get<LocalPortID>(in_port).local_port_id, \
            //         ctx.get<GlobalPortID>(in_port).global_port_id);

            ctx.get<SimTime>(in_port).sim_time = 0;
            ctx.get<SimTimePerUpdate>(in_port).sim_time_per_update = LOOKAHEAD_TIME; //1000ns

            ctx.data().inPorts[ctx.data().numInPort++] = in_port;
        }
    }

    // ingress_ports for core switch
    for (uint32_t i = 0; i < num_core_switch; i++) {
        for (uint32_t j = 0; j < k_ary; j++) {
            Entity in_port = ctx.makeEntity<IngressPort>();
            ctx.get<PortType>(in_port) = PortType::InPort;
            ctx.get<LocalPortID>(in_port).local_port_id = j;
            ctx.get<GlobalPortID>(in_port).global_port_id = (num_edge_switch+num_aggr_switch+i)*k_ary+j;

            ctx.get<PktBuf>(in_port).head = 0;
            ctx.get<PktBuf>(in_port).tail = 0;
            ctx.get<PktBuf>(in_port).cur_num = 0;
            ctx.get<PktBuf>(in_port).cur_bytes = 0;

            ctx.get<SwitchID>(in_port).switch_id = i + num_edge_switch + num_aggr_switch;

            ctx.get<SimTime>(in_port).sim_time = 0;
            ctx.get<SimTimePerUpdate>(in_port).sim_time_per_update = LOOKAHEAD_TIME; //1000ns

            ctx.data().inPorts[ctx.data().numInPort++] = in_port;
        }
    }

    printf("Total %d switches, %d in_ports, %d e_ports\n", \
           ctx.data().numSwitch, ctx.data().numInPort, ctx.data().numEPort);
}


void generate_e_port(Engine &ctx, CountT k_ary)
{
    CountT num_edge_switch = k_ary*k_ary/2;
    CountT num_aggr_switch = k_ary*k_ary/2; // the number of aggragate switch
    CountT num_core_switch = k_ary*k_ary/4;

    printf("*******Generate egress ports*******\n");
    // egress_ports for edge switch

    ctx.data().numEPort = 0;
    for (uint32_t i = 0; i < num_edge_switch; i++) {
        for (uint32_t j = 0; j < k_ary; j++) {
            Entity e_port = ctx.makeEntity<EgressPort>();
            ctx.get<SchedTrajType>(e_port) = SchedTrajType::SP;
            ctx.get<PortType>(e_port) = PortType::EPort;
            ctx.get<LocalPortID>(e_port).local_port_id = j;
            ctx.get<GlobalPortID>(e_port).global_port_id = i*k_ary+j; //i*k_ary+j;

            //pfc
            for (uint32_t k = 0; k < QUEUE_NUM; k++) {
                ctx.get<PktQueue>(e_port).pkt_buf[k].head = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].tail = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].cur_num = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].cur_bytes = 0;

                ctx.get<PktQueue>(e_port).queue_pfc_state[k] = PFCState::RESUME;
            }

            ctx.get<TXHistory>(e_port).head = 0;
            ctx.get<TXHistory>(e_port).tail = 0;
            ctx.get<TXHistory>(e_port).cur_num = 0;
            ctx.get<TXHistory>(e_port).cur_bytes = 0;

            ctx.get<SwitchID>(e_port).switch_id = i;

            if(j < k_ary/2) 
                ctx.get<NextHopType>(e_port) = NextHopType::HOST;
            else
                ctx.get<NextHopType>(e_port) = NextHopType::SWITCH;

            uint32_t ett_idx = ctx.data().numEPort;
            ctx.get<NextHop>(e_port).next_hop = next_hop_table[ett_idx]; // next_hop_table also include mapping  switch to host nic

            ctx.get<LinkRate>(e_port).link_rate = 1000LL*1000*1000*100;
            ctx.get<SSLinkDelay>(e_port).SS_link_delay = SS_LINK_DELAY;

            ctx.get<SimTime>(e_port).sim_time = 0;
            ctx.get<SimTimePerUpdate>(e_port).sim_time_per_update = LOOKAHEAD_TIME; //1000ns

            ctx.get<Seed>(e_port).seed = i+1;

            ctx.data().ePorts[ctx.data().numEPort++] = e_port;
        }
    }

    // egress_ports for aggragate switch
    for (uint32_t i = 0; i < num_aggr_switch; i++) {
        for (uint32_t j = 0; j < k_ary; j++) {
            Entity e_port = ctx.makeEntity<EgressPort>();
            ctx.get<SchedTrajType>(e_port) = SchedTrajType::SP;
            ctx.get<PortType>(e_port) = PortType::EPort;
            ctx.get<LocalPortID>(e_port).local_port_id = j; //i*k_ary+j;
            ctx.get<GlobalPortID>(e_port).global_port_id = (i+num_edge_switch)*k_ary+j;

            //pfc
            for (uint32_t k = 0; k < QUEUE_NUM; k++) {
                ctx.get<PktQueue>(e_port).pkt_buf[k].head = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].tail = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].cur_num = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].cur_bytes = 0;

                ctx.get<PktQueue>(e_port).queue_pfc_state[k] = PFCState::RESUME; 
            }

            ctx.get<TXHistory>(e_port).head = 0;
            ctx.get<TXHistory>(e_port).tail = 0;
            ctx.get<TXHistory>(e_port).cur_num = 0;
            ctx.get<TXHistory>(e_port).cur_bytes = 0;

            ctx.get<SwitchID>(e_port).switch_id = i + num_edge_switch;

            ctx.get<NextHopType>(e_port) = NextHopType::SWITCH;

            uint32_t ett_idx = ctx.data().numEPort;
            ctx.get<NextHop>(e_port).next_hop = next_hop_table[ett_idx];

            ctx.get<LinkRate>(e_port).link_rate = 1000LL*1000*1000*100;
            ctx.get<SSLinkDelay>(e_port).SS_link_delay = SS_LINK_DELAY;

            ctx.get<SimTime>(e_port).sim_time = 0;
            ctx.get<SimTimePerUpdate>(e_port).sim_time_per_update = LOOKAHEAD_TIME; //1000ns

            ctx.get<Seed>(e_port).seed = i+1;

            ctx.data().ePorts[ctx.data().numEPort++] = e_port;
        }
    }

    // egress_ports for core switch
    for (uint32_t i = 0; i < num_core_switch; i++) {
        for (uint32_t j = 0; j < k_ary; j++) {

            Entity e_port = ctx.makeEntity<EgressPort>();
            ctx.get<SchedTrajType>(e_port) = SchedTrajType::SP;
            ctx.get<PortType>(e_port) = PortType::EPort;
            ctx.get<LocalPortID>(e_port).local_port_id = j; //i*k_ary+j;
            ctx.get<GlobalPortID>(e_port).global_port_id = ctx.data().numEPort + (i+num_edge_switch+num_aggr_switch)*k_ary+j;

            //pfc
            for (uint32_t k = 0; k < QUEUE_NUM; k++) {
                ctx.get<PktQueue>(e_port).pkt_buf[k].head = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].tail = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].cur_num = 0;
                ctx.get<PktQueue>(e_port).pkt_buf[k].cur_bytes = 0;

                ctx.get<PktQueue>(e_port).queue_pfc_state[k] = PFCState::RESUME; 
            }

            ctx.get<TXHistory>(e_port).head = 0;
            ctx.get<TXHistory>(e_port).tail = 0;
            ctx.get<TXHistory>(e_port).cur_num = 0;
            ctx.get<TXHistory>(e_port).cur_bytes = 0;

            ctx.get<SwitchID>(e_port).switch_id = i + num_edge_switch + num_aggr_switch;
            
            ctx.get<NextHopType>(e_port) = NextHopType::SWITCH;

            uint32_t ett_idx = ctx.data().numEPort;
            ctx.get<NextHop>(e_port).next_hop = next_hop_table[ett_idx];

            ctx.get<LinkRate>(e_port).link_rate = 1000LL*1000*1000*100;
            ctx.get<SSLinkDelay>(e_port).SS_link_delay = SS_LINK_DELAY;

            ctx.get<SimTime>(e_port).sim_time = 0;
            ctx.get<SimTimePerUpdate>(e_port).sim_time_per_update = LOOKAHEAD_TIME; //1000ns

            ctx.get<Seed>(e_port).seed = i+1;

            ctx.data().ePorts[ctx.data().numEPort++] = e_port;
        }
    }
    printf("Total %d switches, %d in_ports, %d e_ports\n", \
           ctx.data().numSwitch, ctx.data().numInPort, ctx.data().numEPort);
}

void generate_host(Engine &ctx, CountT k_ary)
{
    CountT num_host = k_ary*k_ary*k_ary/4;

    printf("*******Generate NPUs*******\n");
    //NPUs
    for (uint32_t i = 0; i < num_host; i++) {
        Entity npu_e = ctx.makeEntity<NPU>();
        
        ctx.get<NPU_ID>(npu_e).npu_id = i;
        
        ctx.get<SimTime>(npu_e).sim_time = 0;
        ctx.get<SimTimePerUpdate>(npu_e).sim_time_per_update = LOOKAHEAD_TIME; //1000ns

        ctx.get<SendFlows>(npu_e).head = 0;
        ctx.get<SendFlows>(npu_e).tail = 0;
        ctx.get<SendFlows>(npu_e).cur_num = 0;

        ctx.get<CompletedFlowQueue>(npu_e).head = 0;
        ctx.get<CompletedFlowQueue>(npu_e).tail = 0;
        ctx.get<CompletedFlowQueue>(npu_e).cur_num = 0;

        ctx.get<NewFlowQueue>(npu_e).head = 0;
        ctx.get<NewFlowQueue>(npu_e).tail = 0;
        ctx.get<NewFlowQueue>(npu_e).cur_num = 0;

        ctx.data()._npus[ctx.data().num_npu++] = npu_e;
    }

    printf("*******Generate NICs*******\n");
    //nic
    for (uint32_t i = 0; i < num_host; i++) {
        // if (i > 3) {
        //     break;
        // }
        // printf("nic: %d\n", i);
        Entity nic_e = ctx.makeEntity<NIC>();
        ctx.get<NIC_ID>(nic_e).nic_id = i;
        // ctx.get<EgressPortID>(nic_e).egress_port_id = i;

        // printf("MountedFlows\n");
        ctx.get<MountedFlows>(nic_e).head = 0;
        ctx.get<MountedFlows>(nic_e).tail = 0;
        ctx.get<MountedFlows>(nic_e).cur_num = 0;

        // printf("BidPktBuf\n");
        ctx.get<BidPktBuf>(nic_e).snd_buf.head = 0;
        ctx.get<BidPktBuf>(nic_e).snd_buf.tail = 0;
        ctx.get<BidPktBuf>(nic_e).snd_buf.cur_num = 0;
        ctx.get<BidPktBuf>(nic_e).snd_buf.cur_bytes = 0;

        ctx.get<BidPktBuf>(nic_e).recv_buf.head = 0;
        ctx.get<BidPktBuf>(nic_e).recv_buf.tail = 0;
        ctx.get<BidPktBuf>(nic_e).recv_buf.cur_num = 0;
        ctx.get<BidPktBuf>(nic_e).recv_buf.cur_bytes = 0;

        // printf("TXHistory\n");
        ctx.get<TXHistory>(nic_e).head = 0;
        ctx.get<TXHistory>(nic_e).tail = 0;
        ctx.get<TXHistory>(nic_e).cur_num = 0;
        ctx.get<TXHistory>(nic_e).cur_bytes = 0;

        // printf("NextHopType\n");
        ctx.get<NextHopType>(nic_e) = NextHopType::SWITCH;

        // printf("HSLinkDelay\n");
        ctx.get<HSLinkDelay>(nic_e).HS_link_delay = HS_LINK_DELAY; // 1 us, 1000 ns 
        // printf("NICRate\n");       
        ctx.get<NICRate>(nic_e).nic_rate = 1000LL*1000*1000*100; // 100 Gbps 
        // printf("ett_idx\n");  
        uint32_t ett_idx = ctx.data().num_nic;
        // printf("NextHop\n");
        ctx.get<NextHop>(nic_e).next_hop = next_hop_table_host_to_sw[ett_idx];
        // printf("SimTime\n");
        ctx.get<SimTime>(nic_e).sim_time = 0;

        ctx.get<Seed>(nic_e).seed = i;
        ctx.get<SimTimePerUpdate>(nic_e).sim_time_per_update = LOOKAHEAD_TIME; //1000ns
        
        ctx.data()._nics[ctx.data().num_nic++] = nic_e;
    }


    memset(ctx.data().flow_count, 0, MAX_PATH_LEN*sizeof(uint32_t));
    // ctx.data().flow_count = 

    // printf("Total %d switches, %d in_ports, %d e_ports, %d send_flows, %d recv_flows\n", \
    //        ctx.data().numSwitch, ctx.data().numInPort, ctx.data().numEPort, ctx.data().num_snd_flow, ctx.data().num_recv_flow);
    
    printf("Total %d switches, %d in_ports, %d e_ports, %d npus\n", \
           ctx.data().numSwitch, ctx.data().numInPort, ctx.data().numEPort, ctx.data().num_npu);
}



}
