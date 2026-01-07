/*
kód a kapacitásos esetre
bemenet:
hasonló mint eddig, csak az srlg-k élfelsorolása után ott van a kapacitás is
kimenet ugyanaz


hiányosságok:

- nagyon kevés tesztre próbáltam ki
- hecker nincs megírva, automatikusan igazzal tér vissza
- int - long long lehet, hogy rossz, 10^9-ról nem biztos, hogy lehet indítani dolgokat
- spfa részt nem ellenõriztem (bár eddig mindig jól mûködött)
- az utak megtalálásánál továbbra is van olyan rész, aminek helyességében nem vagyok biztos
- lényegtelen kommentek bele vannak írva és lehet, hogy fontosak hiányozank
*/

// g++ dateline_prob.cpp -o dateline_prob -std=c++11
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <list>
#include <limits>
#include <set>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>



using namespace std;

// set max values
const int Max_Node_Count=1000005, Max_Edge_Count=4*Max_Node_Count, Max_Dual_Node_Count=Max_Edge_Count, inf=1000000000;

// Nodes
int Node_Count; // number of nodes
int Node_Id[Max_Node_Count]; // We store the node IDs in reading order, and work with the indices.
//Start indexing from 1, so there it's easier to find if a data is set or not.
int X_Coord[Max_Node_Count], Y_Coord[Max_Node_Count]; // X and Y coordinates of nodes in the same order as the IDs
int Starting_Node, Finishing_Node;
map<int, int> Node_Id_Inverse; // maps node IDs to indices

// Edges
int Edge_Count, Primal_Edge_Left_Endpoint[Max_Edge_Count], Primal_Edge_Right_Endpoint[Max_Edge_Count], Primal_Edge_Id[Max_Edge_Count];
int Next_Node_Left_Fixed[Max_Edge_Count], Next_Node_Right_Fixed[Max_Edge_Count]; // contains the next neighbor of the left and right endpoints
int Prev_Node_Left_Fixed[Max_Edge_Count], Prev_Node_Right_Fixed[Max_Edge_Count];
map<int, int> Edge_Id_Inverse; // maps edge IDs to indices
map<pair<int, int>, int> Edge_Inverse; // maps edge endpoints to indices
map<int, int> srlg_id_inverse; // maps SRLG ids to indices
map<int, int> srlg_ids; // maps SRLG indices to ids

int Primal_Edge_Left_Side[Max_Edge_Count], Primal_Edge_Right_Side[Max_Edge_Count]; // contains the ids of the primal areas
int Primal_Area_Count = 0; // number of different primal areas (dual nodes)
vector<set<int>> Primal_Area_Boundary;
int Primal_Edge_Weight[Max_Edge_Count];
bool is_Special_Edge[Max_Edge_Count];

vector<int> Primal_Adj_List[Max_Node_Count]; // i-th element stores the neighbours of the i-th node
vector<int> SRLG_list[Max_Edge_Count]; // i-th element stores the srlg ids that contains edge i
vector<int> Special_st_Path; // stores a BFS shortest-path
vector<int> srlg_capacities = {}; // stores the capacities of the SRLGs, indexed from Primal_Area_Count+1
vector<double> srlg_probabilities = {}; // stores the probabilities of the SRLGs, indexed from Primal_Area_Count+1
vector<int> final_potential; // i_th element contains the potential of the i-th node and its neighbours

bool check_input=true;
bool check_output=true;

int Answer_Upper_Bound=inf, SRLG_Sum=0;

struct weighted_edge {
    int dest, weight, edge_id;
};


int Dual_Node_Count = 0; // = Primal_Area_Count + number of SRLGs
// format: destination, weight, edge_id
vector<weighted_edge> Dual_Adj_List[Max_Dual_Node_Count];
int SRLG_Count;
vector<vector<int>> srlg_edges = {{}}; // srlg_edges[i] stores the edge ids of the i-th SRLG
vector<vector<int>> SRLG_Disjoint_Paths; // SRLG_Disjoint_Paths[i] stores the i-th SRLG disjoint path
vector<vector<vector<int>>> all_paths = {}; // i-th element stores i paths
vector<map<int, int>> all_srlg_capacities = {}; // i-th element stores the capacities of srlgs correspoinding to i paths
vector<vector<int>> next_edge_in_path;
string output = "";

// needed for negative cycle detection
vector<int> parents; // parents of the nodes. for the root node it will be 0, if there is no negative cycle. If the node is not reachable, then it will be -1
vector<int> negative_cycle; // stores the SRGL ids of the negative cycle
vector<vector<int> > negative_cycle_srlgs; // stores the srlg-edges of the negative cycle

/**
 * @brief Prints the elements of a vector of integers.
 * 
 * This function takes a vector of integers as input and prints each element
 * followed by a space. After printing all elements, it prints a newline character.
 * 
 * @param v The vector of integers to be printed.
 */
void print_vector(vector<int> v) {

    for (auto x:v) {
        cout << x << " ";
    }
    cout << endl;
}

/**
 * @brief Reads an integer from standard input, removing any non-digit characters from the end.
 * 
 * This function reads a string from the standard input, removes any trailing non-digit characters,
 * and then converts the remaining characters to an integer. If the number is negative, it handles
 * the negative sign appropriately.
 * 
 * @return int The integer value read from the input.
 */
int read() {
    
    string s;
    cin >> s;
    // remove the non-digit characters from the end
    while (s.back()>'9' || s.back()<'0') {
        s.pop_back();
    }

    int res=0, po=1; //res is the result, po is the power of 10
    // read the number from the end
    while ('0'<=s.back() && s.back()<='9') {
        res+=po*(s.back()-'0');
        po*=10;
        s.pop_back();
    }

    // if the number is negative, multiply it by -1
    if (s.size()>0 && s.back()=='-') {
        res*=-1;
    }

    return res;

}

/**
 * @brief Prints the total number of SRLG disjoint paths, each path's node ID.
 * 
 * This function performs the following tasks:
 * 1. Prints the total number of SRLG disjoint paths.
 * 2. Iterates through each SRLG disjoint path, prints the node IDs, and calculates the path length.
 * 
 * The function assumes the existence of the following global variables:
 * - SRLG_Disjoint_PPaths: A vector of vectors, where each inner vector represents an SRLG disjoint path.
 * - Node_Id: A map or array that translates node indices to their corresponding IDs.
 */
void print_the_answer() {

    long double total_len=0; // total length of all SRLG disjoint paths

    // print the total number of SRLG disjoint paths
    cout << "number of paths: " << SRLG_Disjoint_Paths.size() << endl;

    // iterate through each SRLG disjoint path, print the node IDs, and calculate the path length
    for (int i=0; i<SRLG_Disjoint_Paths.size(); i++) { 
        int length=0;
        for (auto x:SRLG_Disjoint_Paths[i]) {
            cout << Node_Id[x] << " ";
            length++;
            total_len++;
        }
        cout << endl;
        total_len--;
    }

    // calculate and print the average length of the SRLG disjoint paths
    // cout << "average length: " << total_len/SRLG_Disjoint_Paths.size() << endl;
}


/**
 * @brief Adds a Shared Risk Link Group (SRLG) to the network.
 * 
 * This function processes a given set of edges that form an SRLG, builds the dual graph representation,
 * checks connectivity, and determines if the SRLG is cutting. If not cutting, the SRLG is added to the
 * dual graph with an extra node.
 * 
 * @param edges A vector containing the IDs of edges that belong to the SRLG.
 * @param SRLG_Capacity The capacity assigned to the SRLG.
 * @param srlg_id The unique identifier for the SRLG.
 * 
 * The function performs the following steps:
 * - Constructs an adjacency list for areas using the given edges.
 * - Uses BFS to check connectivity of the SRLG in the area graph.
 * - Identifies if the SRLG is "cutting" (i.e., it creates a cycle with inconsistent distances).
 * - If the SRLG is connected and non-cutting, it is added to the dual graph with an extra node.
 * - Updates SRLG-related data structures, including capacities and edge assignments.
 * 
 * @note If the SRLG is not connected, the program prints an error message and exits.
 * @note If the SRLG is cutting and should be eliminated, it is not added to the dual graph.
 * 
 * Global Variables Used:
 * - `Primal_Edge_Left_Side`, `Primal_Edge_Right_Side`, `Primal_Edge_Weight`: Define the edges in the primal graph.
 * - `Primal_Area_Count`: The total number of areas.
 * - `SRLG_Sum`: Stores the total SRLG capacity.
 * - `Answer_Upper_Bound`: Stores the upper bound on the solution.
 * - `Dual_Node_Count`: Tracks the number of nodes in the dual graph.
 * - `Dual_Adj_List`: Adjacency list representation of the dual graph.
 * - `SRLG_list`: Stores the SRLGs associated with each edge.
 * - `SRLG_Capacities`: Maps dual nodes to their assigned SRLG capacities.
 */
void add_SRLG(vector<int> edges, int srlg_capacity, double srlg_probability, int srlg_id) {
    // this is an array of vectors. vector at index i contains the neighbours of the i-th area
    // the neighbours are stored as pairs of (area_id, weight)
    vector<vector<pair<int, int>>> Area_Adj_List(Primal_Area_Count + 1);
    
    vector<int> Dist(Primal_Area_Count+1, 0);
    set<int> Important_Areas;
    
    int st = 0;

    // iterate through each edge in the SRLG
    for (auto Edge_id:edges) {

        int a=Primal_Edge_Left_Side[Edge_id];
        int b=Primal_Edge_Right_Side[Edge_id];
        int c=Primal_Edge_Weight[Edge_id]; // weight of the edge. left -> right is positive, right -> left is negative

        // add all areas that should be connected
        Important_Areas.insert(a), Important_Areas.insert(b);
        // Build the dual graph. left -> right is positive, right -> left is negative
        Area_Adj_List[a].push_back({b, c}), Area_Adj_List[b].push_back({a, -c});
        if (st == 0) { // set starting area
            st = a;
        }
    }

    // BFS for calculating Dist, checking srlg connectivity and check if the srlg is cutting
    vector<bool> Is_Visited(Primal_Area_Count+1, 0);
    queue<int> q;
    int vis_count = 0; // number of visited areas, so we can check connectivity

    q.push(st); // in q, we store the area ids
    Is_Visited[st]=1;

    // go through the areas
    while (q.size()>0) {
        // get the oldest area id, and delete it from the queue
        int Cur_Area=q.front(); 
        q.pop();

        vis_count++; // count number of visited areas
        // go through the neighbours of the current area
        for (auto x:Area_Adj_List[Cur_Area]) {
            // get the current neighbor's id and the distance from the current area
            int Next_Area=x.first, Next_Dist=Dist[Cur_Area]+x.second;
            // if the area is not visited, add it to the queue, and set the distance
            if (!Is_Visited[Next_Area]) {
                Is_Visited[Next_Area] = 1;
                Dist[Next_Area] = Next_Dist;
                q.push(Next_Area);
            }
        }
    }

    SRLG_Sum+=srlg_capacity;

    if (vis_count!=Important_Areas.size()) { // check if the SRLG is connected
        cerr << "Wrong input: SRLG " << srlg_id << " " << "is not connected" << endl;
        cerr << "SRLG edges: ";
        for (auto edge : edges) {
            cerr << edge << " ";
        }
        cerr << endl;
        exit(1);
    }
    Dual_Node_Count++; // new node index
    for (auto x:Important_Areas) {
        // add edges from the areas to the new node (virtual edges with id 0)
        Dual_Adj_List[x].push_back({Dual_Node_Count, -Dist[x], 0});
        // add edges from the new node to the areas
        Dual_Adj_List[Dual_Node_Count].push_back({x, Dist[x], srlg_id});
    }
    for (auto x:edges) {
        // for all edges in the SRLG, add the srlg_id to the edge's list
        SRLG_list[x].push_back(srlg_id);
    }
    // add the SRLG capacity to the extra node
    srlg_id_inverse[srlg_id] = Dual_Node_Count;
    srlg_ids[Dual_Node_Count] = srlg_id;
    srlg_capacities.push_back(srlg_capacity);
    srlg_probabilities.push_back(srlg_probability);
}

/**
 * @brief Checks for duplicate node IDs and coordinates.
 * 
 * This function iterates through all nodes and checks if there are any nodes
 * with the same ID or the same coordinates. If it finds any duplicates, it 
 * prints an error message and terminates the program.
 * 
 * @details
 * The function uses two nested loops to compare each node with every other node.
 * If two nodes have the same ID, it prints an error message indicating the node
 * indices and exits the program. Similarly, if two nodes have the same coordinates,
 * it prints the coordinates and an error message indicating the node indices, then
 * exits the program.
 * 
 * @note
 * The function assumes that Node_Count, Node_Id, X_Coord, and Y_Coord are defined
 * and accessible within the scope of this function.
 */
void check_nodes() {

    for (int i=1; i<=Node_Count; i++) {
        for (int j=i+1; j<=Node_Count; j++) {

            // check for duplicate node IDs
            if (Node_Id[i]==Node_Id[j]) {
                cout << "Wrong input: same id for node " << i << " and node " << j << endl;
                exit(0);
            }

            // check for duplicate coordinates
            if (X_Coord[i]==X_Coord[j] && Y_Coord[i]==Y_Coord[j]) {
                cout << X_Coord[i] << " " << X_Coord[j] << " " << Y_Coord[i] << " " << Y_Coord[j] << endl;
                cout << "Wrong input: same coordinates for node " << i << " and node " << j << endl;
                exit(0);
            }
        }
    }
}

/**
 * @brief Reads the nodes and their coordinates from input.
 * 
 * This function reads the total number of nodes, their coordinates, and their IDs.
 * It also reads the starting and finishing nodes for a pathfinding or graph-related algorithm.
 * 
 * The function performs the following steps:
 * 1. Reads the total number of nodes and asserts that there are at least two nodes.
 * 2. Reads the coordinates (X and Y) and IDs for each node, storing them in arrays.
 * 3. Maps each node ID to its corresponding index.
 * 4. Reads the starting and finishing nodes and maps them to their corresponding indices.
 * 5. Asserts that the starting and finishing nodes are valid.
 * 6. Optionally checks the validity of the nodes if the `check_input` flag is set.
 * 
 * @note The function assumes that the `read` function is defined elsewhere and is used to read input values.
 * @note The function uses global variables: `Node_Count`, `X_Coord`, `Y_Coord`, `Node_Id`, `Node_Id_Inverse`, `Starting_Node`, `Finishing_Node`, and `check_input`.
 * @note The function calls `check_nodes` if `check_input` is true.
 */
void read_nodes() {
    Node_Count = read();
    assert(Node_Count > 1); // we need at least two nodes

    // read the nodes
    for (int i=1; i<=Node_Count; i++) {
        X_Coord[i]=read(), Y_Coord[i]=read(), Node_Id[i]=read();
        Node_Id_Inverse[Node_Id[i]]=i;
    }

    // read the starting and finishing nodes
    Starting_Node=read(), Finishing_Node=read();
    Starting_Node=Node_Id_Inverse[Starting_Node], Finishing_Node=Node_Id_Inverse[Finishing_Node];

    // check if the input is valid
    assert(1<=Starting_Node && Starting_Node<=Node_Count && 1<=Finishing_Node && Finishing_Node<=Node_Count);
    if (check_input) {
        check_nodes();
    }
}

/**
 * @brief Calculates the orientation of three points.
 * 
 * This function calculates the orientation of three points using the cross product.
 * It takes three node indices as input and returns the orientation as an integer.
 * 
 * @param a The index of the first node.
 * @param b The index of the second node.
 * @param c The index of the third node.
 * @return int The orientation of the points: 1 for a left turn, 0 for collinear, -1 for a right turn.
 */
int direction(int a, int b, int c) {
    assert(1<=min({a, b, c}) && max({a, b, c})<=Node_Count); // valid node indices
    assert(a!=b && a!=c && b!=c); // different nodes

    // calculate the orientation of the points using the cross product
    long long val=1ll*(X_Coord[b]-X_Coord[a])*(Y_Coord[c]-Y_Coord[a])-1ll*(X_Coord[c]-X_Coord[a])*(Y_Coord[b]-Y_Coord[a]);
    
    // return the orientation: 1 for left turn, 0 for collinear, -1 for right turn
    if (val>0) return 1;
    else if (val==0) return 0;
    else return -1;
}

/**
 * @brief Checks if a point is between two other points.
 * 
 * This function checks if a point is between two other points along the x-axis or y-axis.
 * It takes three node indices as input and returns a boolean value indicating if the point is between the other two.
 * 
 * @param a The index of the point to check.
 * @param b The index of the first point.
 * @param c The index of the second point.
 * @return bool True if the point is between the other two, false otherwise.
 */
bool is_between(int a, int b, int c) {
    assert(direction(a, b, c)==0); // collinear points
    // check if point a is between points b and c
    if ((X_Coord[a]<X_Coord[b])+(X_Coord[a]<X_Coord[c])==1) return true; // XOR
    if ((Y_Coord[a]<Y_Coord[b])+(Y_Coord[a]<Y_Coord[c])==1) return true; // XOR
    return false;
}

/**
 * @brief Checks the planarity of a graph by verifying that no two edges intersect.
 * 
 * This function iterates through all pairs of edges in the graph and checks for intersections.
 * It handles three cases:
 * 1. Two endpoints of the edges are equal, indicating an intersection.
 * 2. One endpoint of the edges is equal, and the other endpoint is checked if it lies between the other two endpoints.
 * 3. No endpoints are equal, and the function checks if the edges intersect using directional checks.
 * 
 * If an intersection is found, the function prints an error message and exits the program.
 * 
 * @note The function assumes that the global variables `Edge_Count`, `Primal_Edge_Left_Endpoint`, 
 * and `Primal_Edge_Right_Endpoint` are defined and properly initialized.
 * 
 * @throws std::runtime_error if an intersection between edges is detected.
 */
void check_planarity() {
    // iterate through all pairs of edges
    for (int i=1; i<=Edge_Count; i++) {
        for (int j=i+1; j<=Edge_Count; j++) {
            int a=Primal_Edge_Left_Endpoint[i], b=Primal_Edge_Right_Endpoint[i];
            int c=Primal_Edge_Left_Endpoint[j], d=Primal_Edge_Right_Endpoint[j];
            int Equal_Count=(a==c)+(a==d)+(b==c)+(b==d);
            assert(Equal_Count<=2); // at most two endpoints can be equal
            // if two endpoints are equal, edges are intersecting
            if (Equal_Count==2) { 
                cout << "Wrong input: intersection between edge " << i << " and edge " << j << endl;
                exit(0);
            }
            // if one endpoint is equal, check if the other endpoint is between the other two endpoints
            if (Equal_Count==1) {
                // swap the endpoints to make a and c equal
                if (b==c || b==d) swap(a, b);
                if (a==d) swap(c, d);
                assert(a==c);
                
                // check if the other endpoint is between the other two endpoints
                if (direction(a, b, d)==0 && !is_between(a, b, d)) {
                    cout << "Wrong input: intersection between edge " << i << " and edge " << j << endl;
                    exit(0);
                }
            }
            // if no endpoints are equal, check if the edges intersect
            if (Equal_Count==0) {
                int dir_abc=direction(a, b, c), dir_abd=direction(a, b, d);
                
                if (dir_abc==0 && dir_abd==0) { // collinear points
                    assert(direction(c, d, a)==0 && direction(c, d, b)==0);
                    if ((is_between(a, c, d) || is_between(b, c, d)) && (is_between(c, a, b) || is_between(d, a, b))) {
                        cout << "Wrong input: intersection between edge " << i << " and edge " << j << endl;
                        exit(0);
                    }
                } else if (dir_abc+dir_abd==0) {
                    int dir_cda=direction(c, d, a), dir_cdb=direction(c, d, b);
                    if (dir_cda+dir_cdb==0) {
                        cout << "Wrong input: intersection between edge " << i << " and edge " << j << endl;
                        exit(0);
                    }
                }
            }
        }
    }
}

/**
 * @brief Reads and processes the edges of a graph.
 *
 * This function reads the total number of edges and iterates through each edge to read its endpoints and ID.
 * It performs several checks to ensure the validity of the input:
 * - Ensures that the edge endpoints are different.
 * - Maps node IDs to indices and checks if they are valid (non-zero).
 * 
 * For each valid edge, it stores the endpoints and ID in the corresponding arrays and updates the adjacency list
 * and inverse mappings.
 * 
 * If the `check_input` flag is set, it calls the `check_planarity` function to verify the planarity of the graph.
 * 
 * @note The function exits the program if any invalid input is detected.
 */
void read_edges() { 
    Edge_Count=read(); // read the total number of edges
    for (int i=1; i<=Edge_Count; i++) {
        int a, b, c; // edge endpoints and ID
        a=read(), b=read(), c=read();
        if (a==b) { // check if the edge has different endpoints
            cout << "Wrong input: edge " << i << " " << "must have different endpoints" << endl;
            exit(0);
        }

        a=Node_Id_Inverse[a], b=Node_Id_Inverse[b]; // map node IDs to indices
        if (!a || !b) { // check if the node indices are non-zero (valid)
            cout << "Wrong input: there is a problem with edge " << i << endl;
            exit(0);
        }

        // store the edge endpoints and ID in the corresponding arrays
        Primal_Edge_Left_Endpoint[i]=a, Primal_Edge_Right_Endpoint[i]=b; // endpoints
        Primal_Adj_List[a].push_back(b), Primal_Adj_List[b].push_back(a); // nodes' adjacency list
        Primal_Edge_Id[i]=c, Edge_Id_Inverse[c]=i; // edge ID and inverse mapping
        Edge_Inverse[{a, b}]=i, Edge_Inverse[{b, a}]=i; // edge endpoints and inverse mapping
    }
    Primal_Edge_Id[0] = -1;
    if (check_input) {
        check_planarity();
    }
}

/**
 * @brief Checks the connectivity of the primal graph using Breadth-First Search (BFS).
 * 
 * This function performs a BFS starting from node 1 to determine if all nodes in the primal graph
 * are reachable. If any node is not reachable, it prints an error message and exits the program.
 * 
 * The primal graph is represented by an adjacency list `Primal_Adj_List` and the number of nodes
 * is given by `Node_Count`.
 * 
 * @note The function assumes that the nodes are numbered from 1 to `Node_Count`.
 * 
 * @warning The program will terminate if the primal graph is found to be disconnected.
 */
void check_primal_connectivity() { 
    // BFS
    vector<bool> Is_Visited(Node_Count+1, 0);
    queue<int> q;
    q.push(1);
    Is_Visited[1] = true;
    while (q.size() > 0) {
        int Cur_Node=q.front();
        q.pop();
        for (auto Next_Node:Primal_Adj_List[Cur_Node]) {
            if (!Is_Visited[Next_Node]) {
                q.push(Next_Node);
                Is_Visited[Next_Node] = true;
            }
        }
    }

    // check if the primal graph is connected
    for (int i=1; i<=Node_Count; i++) {
        if (!Is_Visited[i]) {
            cout << "Wrong input: the primal graph is not connected" << endl;
            exit(0);
        }
    }
}

/**
 * @brief Orders the neighbours of each node in a graph in clockwise order.
 * 
 * This function iterates through each node in the graph and sorts its neighbours
 * based on the angle they make with the node, in a clockwise direction. It then
 * updates the adjacency list of each node with this new order. Additionally, it
 * updates the fixed neighbours for each edge based on the new order.
 * 
 * The function performs the following steps for each node:
 * 1. Calculates the angle for each neighbour and stores the neighbour index.
 * 2. Sorts the neighbours in clockwise order based on the calculated angles.
 * 3. Updates the adjacency list of the node with the new order.
 * 4. Updates the fixed neighbours for each edge based on the new order.
 * 
 * The function assumes the following global variables are defined:
 * - Node_Count: The total number of nodes in the graph.
 * - Primal_Adj_List: The adjacency list of the graph.
 * - X_Coord, Y_Coord: The coordinates of each node.
 * - Edge_Inverse: A map from edge pairs to edge IDs.
 * - Primal_Edge_Left_Endpoint, Primal_Edge_Right_Endpoint: The endpoints of each edge.
 * - Next_Node_Left_Fixed, Next_Node_Right_Fixed: The next fixed neighbours for each edge.
 * - Prev_Node_Left_Fixed, Prev_Node_Right_Fixed: The previous fixed neighbours for each edge.
 */
void order_neighbours() { 
    // sort the neighbours of each node in clockwise order
    for (int i=1; i<=Node_Count; i++) {
        vector<pair<long double, int> > New_Order;

        // for each neighbour of the current node, calculate the angle and store the neighbour index
        for (auto adj:Primal_Adj_List[i]) {
            New_Order.push_back({atan2(X_Coord[adj]-X_Coord[i], Y_Coord[adj]-Y_Coord[i]), adj});
        }
        // sort the neighbours in clockwise order
        sort(New_Order.begin(), New_Order.end());

        // update the adjacency list with the new order
        Primal_Adj_List[i].clear();
        for (auto adj:New_Order) {
            Primal_Adj_List[i].push_back(adj.second);
        }

        // update the fixed neighbours for each edge
        for (int j=0; j<Primal_Adj_List[i].size(); j++) {

            int a=i; // current node
            int b=Primal_Adj_List[i][j]; // j-th neighbor
            int c=Primal_Adj_List[i][(j+1) % Primal_Adj_List[i].size()]; // next neighbor

            // consider edge (a,b)
            int edge_id=Edge_Inverse[{a, b}];
            // if a is left endpoint, then c is the next left-node
            if (Primal_Edge_Left_Endpoint[edge_id]==a) {
                Next_Node_Left_Fixed[edge_id]=c;
            // if a is right endpoint, then c is the next right-node
            } else {
                assert(Primal_Edge_Right_Endpoint[edge_id]==a);
                Next_Node_Right_Fixed[edge_id]=c;
            }

            // similarly for edge (a,b)
            edge_id=Edge_Inverse[{a, c}];
            if (Primal_Edge_Left_Endpoint[edge_id]==a) {
                Prev_Node_Left_Fixed[edge_id]=b;
            } else {
                assert(Primal_Edge_Right_Endpoint[edge_id]==a);
                Prev_Node_Right_Fixed[edge_id]=b;
            }

        }
    }
}

/**
 * @brief Finds the next node in order based on the given edge (x,y).
 *
 * This function takes two node identifiers, `x` and `y`, and determines the next node in the order
 * based on the edge between them. It uses the `Edge_Inverse` map to find the edge ID and checks
 * the endpoints of the edge to determine the next node.
 *
 * @param x The first node identifier.
 * @param y The second node identifier.
 * @return The identifier of the next node in order.
 *
 * @pre The edge between `x` and `y` must exist and be within the valid range of edge IDs.
 * @pre `1 <= edge_id <= Edge_Count`
 */
int find_next_in_order(int x, int y) { 
    int edge_id=Edge_Inverse[{x, y}];
    assert(1<=edge_id && edge_id<=Edge_Count);

    if (Primal_Edge_Left_Endpoint[edge_id]==y) {
        return Next_Node_Left_Fixed[edge_id];
    } else {
        return Next_Node_Right_Fixed[edge_id];
    }
}

/**
 * @brief Finds the previous node in order for a given edge (x,y).
 *
 * This function takes two node identifiers, `x` and `y`, and returns the previous node
 * in order based on the edge direction. It uses the `Edge_Inverse` map to find the edge ID
 * and checks if the left endpoint of the edge is `y`. Depending on the endpoint, it returns
 * the previous node from either `Prev_Node_Left_Fixed` or `Prev_Node_Right_Fixed`.
 *
 * @param x The first node identifier.
 * @param y The second node identifier.
 * @return The previous node in order for the given edge.
 * @throws std::assert if the edge ID is not within the valid range.
 */
int find_prev_in_order(int x, int y) { 
    int edge_id=Edge_Inverse[{x, y}];
    assert(1<=edge_id && edge_id<=Edge_Count);
    if (Primal_Edge_Left_Endpoint[edge_id]==y) {
        return Prev_Node_Left_Fixed[edge_id];
    } else {
        return Prev_Node_Right_Fixed[edge_id];
    }
}

/**
 * @brief Finds and sets the primal area for a given edge.
 *
 * This function determines which side of the edge is the right side and sets the corresponding side
 * with the provided id using Primal_Edge_Left_Side, and Primal_Edge_Right_side. 
 * It then recursively processes the next edge in order.
 *
 * @param x The x-coordinate of the edge.
 * @param y The y-coordinate of the edge.
 * @param id The identifier to set for the primal area.
 *
 * @note The function assumes that the edge validation and index retrieval are successful.
 *       It uses assertions to ensure that the coordinates are within valid bounds and that the edge exists.
 */
void find_primal_area(int x, int y, int id) { 
    // edge validation and get index
    assert(1<=x && x<=Edge_Count && 1<=y && y<=Edge_Count);
    int edge=Edge_Inverse[{x, y}];
    assert(edge!=0);

    // decide which side of the edge is the right side
    bool is_right_side = Primal_Edge_Left_Endpoint[edge] != x;

    if (is_right_side) {
        // if the right side is already set, we finished the area
        if (Primal_Edge_Right_Side[edge]) {
            return;
        }
        // else, set the right side
        Primal_Edge_Right_Side[edge]=id;
    }
    else {
        // if the left side is already set, return
        if (Primal_Edge_Left_Side[edge]) {
            return;
        }
        // else, set the left side
        Primal_Edge_Left_Side[edge]=id;
    }

    // find the next edge in order
    int z=find_next_in_order(x, y);
    // call the function recursively
    find_primal_area(y, z, id);
}

/**
 * @brief Finds and counts the primal areas in a graph based on its edges.
 * 
 * This function iterates through each edge in the graph and checks if the left
 * and right sides of the edge have been set. If not, it finds the corresponding
 * primal area and increments the area count. It also sets unique IDs to primal areas.
 * The function also updates the dual node count to match the primal area count.
 * 
 * @note This function assumes that the following global variables and arrays 
 * are defined and properly initialized:
 * - Edge_Count: The total number of edges in the graph.
 * - Primal_Edge_Left_Side: An array indicating whether the left side of each edge is set.
 * - Primal_Edge_Right_Side: An array indicating whether the right side of each edge is set.
 * - Primal_Edge_Left_Endpoint: An array containing the left endpoints of each edge.
 * - Primal_Edge_Right_Endpoint: An array containing the right endpoints of each edge.
 * - Primal_Area_Count: A counter for the number of primal areas found.
 * - Dual_Node_Count: A counter for the number of dual nodes, which is set to the number of primal areas.
 * 
 * The function relies on the helper function `find_primal_area(int x, int y, int area_count)`
 * to recursively find and mark the primal area starting from the given endpoints.
 */
void find_areas() { 
    for (int i=1; i<=Edge_Count; i++) { // iterate through each edge
        // if the left side is not set, find the primal area, and increase the area count
        if (!Primal_Edge_Left_Side[i]) {
            Primal_Area_Count++;
            int x=Primal_Edge_Left_Endpoint[i], y=Primal_Edge_Right_Endpoint[i];
            find_primal_area(x, y, Primal_Area_Count);
        }
        // same for the left side
        if (!Primal_Edge_Right_Side[i]) {
            Primal_Area_Count++;
            int x=Primal_Edge_Right_Endpoint[i], y=Primal_Edge_Left_Endpoint[i];
            find_primal_area(x, y, Primal_Area_Count);
        }
    }
    Dual_Node_Count=Primal_Area_Count;
}

/**
 * @brief Finds the shortest path from the starting node to the finishing node using BFS.
 * 
 * This function performs a breadth-first search (BFS) to find the shortest path from the 
 * starting node to the finishing node in an unweighted graph. It then sets this path as 
 * the special st path and prints it.
 * 
 * @details
 * - Initializes a visited vector and a previous node vector to keep track of the BFS traversal.
 * - Uses a queue to perform BFS starting from the starting node.
 * - For each node, it explores its adjacent nodes and updates the visited and previous node vectors.
 * - After BFS completes, it reconstructs the shortest path from the finishing node to the starting node.
 * - Sets the reconstructed path as the special st path and prints it.
 * 
 * @param None
 * @return void
 */
void find_special_st_path() { 
    vector<bool> Is_Visited(Node_Count+1, 0);
    vector<int> Previous_Node(Node_Count+1, 0);
    queue<int> q;
    q.push(Starting_Node);
    Is_Visited[Starting_Node]=1;

    // BFS
    while (q.size()>0) {
        int Cur_Node=q.front();
        q.pop();
        for (auto Next_Node:Primal_Adj_List[Cur_Node]) {
            if (!Is_Visited[Next_Node]) {
                q.push(Next_Node);
                Is_Visited[Next_Node]=1;
                Previous_Node[Next_Node]=Cur_Node;
            }
        }
    }

    // find the shortest path
    vector<int> st_path;
    int pos=Finishing_Node;
    while (pos!=Starting_Node) {
        st_path.push_back(pos);
        pos=Previous_Node[pos];
    }
    st_path.push_back(Starting_Node);
    reverse(st_path.begin(), st_path.end());

    // set the special st path
    Special_st_Path=st_path;
    // print the special st path
    cout << "first st path:" << endl;
    for (auto x:st_path) {
        cout << Node_Id[x] << " ";
    }
    cout << endl;
}

/**
 * @brief Processes the special st path and sets the weights of the edges accordingly.
 * 
 * This function iterates through the special st path and sets the weights of the edges
 * based on the direction of the path. If the edge is traversed from the left endpoint
 * to the right endpoint, the weight is set to 1. If the edge is traversed from the right
 * endpoint to the left endpoint, the weight is set to -1.
 * 
 * @pre The special st path (Special_st_Path) and the edge inverse map (Edge_Inverse) must be initialized.
 * @pre The primal edge left endpoints (Primal_Edge_Left_Endpoint) and the primal edge weights (Primal_Edge_Weight) must be initialized.
 * @pre The edge count (Edge_Count) must be initialized.
 * 
 * @post The weights of the edges in the primal edge weight array (Primal_Edge_Weight) are updated based on the direction of the path.
 */
void process_the_st_path() { 
    for (int i=1; i<Special_st_Path.size(); i++) { // iterate through the special st path (edges)
        // get the edge ID
        int a=Special_st_Path[i-1], b=Special_st_Path[i], edge_id=Edge_Inverse[{a, b}];
        assert(1<=edge_id && edge_id<=Edge_Count);

        // set edge weights based on the direction of the path.
        if (a==Primal_Edge_Left_Endpoint[edge_id]) {
            Primal_Edge_Weight[edge_id]=1; // left -> right: weight = 1
        } else {
            Primal_Edge_Weight[edge_id]=-1; // right -> left: weight = -1
        }
    }
}

/**
 * @brief Reads the SRLGs (Shared Risk Link Groups) from the standard input.
 * 
 * This function reads the total number of SRLGs and iterates through each SRLG to read its size and edges.
 * If the size of an SRLG is zero, it throws an error and exits the program.
 * Finally, it adds the SRLG to the list using the `add_SRLG` function.
 * Gives each SRLG a unique ID starting from 1.
 * 
 * @note The function expects the input in a specific format:
 *       - First, the total number of SRLGs.
 *       - For each SRLG:
 *         - The size of the SRLG.
 *         - The edges in the SRLG.
 * 
 * @throws If the size of an SRLG is zero or if there is a problem with an edge ID, 
 *         the function prints an error message and exits the program.
 */
void read_SRLGs() { 
    cin >> SRLG_Count; // read the total number of SRLGs

    // Fill with -1s up to the current Dual_Node_Count (primal areas)
    srlg_capacities.assign(Dual_Node_Count+1, -1);
    srlg_probabilities.assign(Dual_Node_Count+1, -1.0);
    
    // iterate through each SRLG
    for (int i=1; i<=SRLG_Count; i++) { 
        vector<int> edges;

        // read the size of the SRLG, throw error is size is 0
        int srlg_size, srlg_capacity = 1;
        double srlg_probability = 1.0;
        int srlg_id = -1;

        cin >> srlg_size;
        if (srlg_size==0) {
            cout << "Wrong input SRLG with 0 size" << endl;
            exit(0);
        }
        
        // read the edges in the SRLG
        for (int j=1; j<=srlg_size; j++) { 
            int Edge_id;
            cin >> Edge_id;
            Edge_id=Edge_Id_Inverse[Edge_id]; // from now, Edge_id is the index of the edge
            if (Edge_id==0) {
                exit(0);
            }
            edges.push_back(Edge_id);
        }

        cin >> srlg_capacity; // read the SRLG capacity
        cin >> srlg_probability; // read the SRLG probability
        cin >> srlg_id; // read the SRLG id
        srlg_id++;

        // add the SRLG to the list
        srlg_edges.push_back(edges);
        add_SRLG(edges, srlg_capacity, srlg_probability, srlg_id);
    }
}

/**
 * @brief Adds dual edges to the dual adjacency list.
 * 
 * This function iterates through each primal edge and adds corresponding dual edges
 * to the dual adjacency list. For each primal edge, it retrieves the left side node,
 * right side node, and weight, then adds two entries to the dual adjacency list:
 * one for each direction with the appropriate weight and edge ID.
 * 
 * The format for each entry in the dual adjacency list is {destination, weight, edge_id}.
 * 
 * @note The edge ID is stored as a negative value to distinguish it from other identifiers.
 */
void add_dual_edges() {
    // iterate through each primal edge
    for (int i=1; i<=Edge_Count; i++) {
        int a=Primal_Edge_Left_Side[i], b=Primal_Edge_Right_Side[i], w=Primal_Edge_Weight[i];
        // format: {dest, weight, edge_id} NOTE: all edges will be non-negative
        Dual_Adj_List[a].push_back({b, w, -i});
        Dual_Adj_List[b].push_back({a, -w, -i});
    }
}

void find_negative_cycle(int start_node, vector<vector<weighted_edge>> &Adj_List) { 
    negative_cycle.clear(); // clear the negative cycle vector
    // get the first node in the negative cycle. This will be the current_node.
    vector<bool> seen(Dual_Node_Count+1, false);
    int current_node = start_node;
    while (!seen[current_node]) {
        seen[current_node] = true;
        current_node = parents[current_node];
    }
    start_node = current_node;
    // now we have the first node in the negative cycle. We need to find the srlg ids that are in the cycle.
    int counter = 0;
    do{
        counter++;
        if (current_node > Primal_Area_Count){ // if the current node a virtual node
            for (weighted_edge edge: Adj_List[current_node]) { // go through its neighbors
                if (edge.dest == parents[current_node]) { // if the edge is in the cycle
                    negative_cycle.push_back(edge.edge_id); // add the edge to the negative cycle
                }
            }
        }
        current_node = parents[current_node]; // go to the next node in the cycle
    } while (current_node != start_node);
}

void find_negative_cycle_srlgs(){
    // go through the negative cycle, and based on the srlg ids, add the srlg edge_id lists.
    cout << "negative cycle:" << endl;
    for (int srlg_id: negative_cycle) {
        negative_cycle_srlgs.push_back(srlg_edges[srlg_id]);
        for (int x: srlg_edges[srlg_id])
              cout << Primal_Edge_Id[x] << " ";
        cout << endl;
    }
}

/**
 * @brief Computes the potential of each node in a graph using a modified SPFA (Shortest Path Faster Algorithm).
 * 
 * This function calculates the potential of each node in a graph represented by an adjacency list. 
 * It uses a queue-based approach to iteratively update the potential values. The function can also 
 * detect negative weight cycles in the graph.
 * 
 * @param n The number of nodes in the graph.
 * @param Adj_List The adjacency list representation of the graph. Each node has a list of pairs, 
 *                 where each pair consists of a neighboring node and the weight of the edge to that neighbor.
 * @return A vector of integers representing the potential of each node. If a negative weight cycle 
 *         is detected, the first element of the vector is set to -1.
 * 
 * @details
 * - The potential of each node is initialized to infinity (except for the starting node, which is set to 0).
 * - The function uses a queue to process nodes and updates their potential values based on the shortest path.
 * - If a negative weight cycle is detected, the function returns immediately with the 0-th element of the 
 *   potential vector set to -1.
 * - The function ensures that no node is processed more than `n` times to avoid infinite loops.
 */
vector<int> find_potential_SPFA(int n, vector<vector<weighted_edge>> &Adj_List, bool find_proof=false) {

    vector<int> Potential(n+1, 0); // this is what we eant to calculate
    vector<int> Opt_Dist(n+1, 0); // number of used nodes during the shorest path
    vector<bool> in_queue(n+1, 0);
    parents.resize(n+1, -1); // parent of each node
    queue<int> q;
    int First_Node=1;
    // first node should be the one with the most neighbours
    for (int i=1; i<=n; i++) { 
        if (Adj_List[i].size()>Adj_List[First_Node].size()) {
            First_Node=i;
        }
    }
    // set the potential for every node to infinity
    int inf=1e9;
    for (int i=1; i<=n; i++) {
        Potential[i]=inf;
    }
    // set the potential for the first node to 0, and push it to the queue
    Potential[First_Node]=0;
    parents[First_Node]=0;
    q.push(First_Node);
    in_queue[First_Node]=true;

    // while the queue is not empty
    while (q.size()>0) {
        // get the oldest element
        int Cur_Node=q.front();

        
        // if there is a negative cycle back to the first node, return
        if (Cur_Node==First_Node && Potential[First_Node]<0) {  // elvileg csak optimalizacio
            if (find_proof) {
                find_negative_cycle(First_Node, Adj_List);
            }
            Potential[0]=-1;
            return Potential;
        }
        // pop the first element
        q.pop();
        in_queue[Cur_Node]=false;
        // if the opt path to this node contains more than n nodes, then there is a negative cycle. return
        if (Opt_Dist[Cur_Node]>n) {
            if (find_proof) {
                find_negative_cycle(Cur_Node, Adj_List);
            }
            Potential[0]=-1;
            return Potential;
        }
        

        // go through the dual nodes
        for (auto x:Adj_List[Cur_Node]) {
            int Next_Node=x.dest, Next_Dist=Potential[Cur_Node]+x.weight, id = x.edge_id;
            // if the potential for the next node is bigger than the new potential, update it
            if (Potential[Next_Node]>Next_Dist) {
                Potential[Next_Node]=Next_Dist;
                Opt_Dist[Next_Node]=Opt_Dist[Cur_Node]+1;
                parents[Next_Node]=Cur_Node;
                // the node is changed, so we need to push it to the queue
                if (!in_queue[Next_Node]) {
                    in_queue[Next_Node]=true;
                    q.push(Next_Node);
                }
            }
        }
    }
    return Potential;
}

/**
 * @brief Finds SRLG (Shared Risk Link Group) disjoint paths in a graph.
 * 
 * This function attempts to find a specified number of SRLG disjoint paths in a graph. 
 * It uses a modified adjacency list with weights and calculates potentials to determine 
 * the feasibility of the paths. If feasible, it constructs the paths while ensuring 
 * disjointness based on SRLG constraints.
 * 
 * @param Val The number of SRLG disjoint paths to find. Must be greater than 0.
 * @param Print_it If true, the function prints the paths; otherwise, it only checks feasibility.
 * @param find_proof If true, the function will return the SRLG ids containing a negative cycle.
 * @return true If the specified number of SRLG disjoint paths is found.
 * @return false If the specified number of SRLG disjoint paths cannot be found.
 * 
 * @details
 * - The function constructs a modified adjacency list with weights based on SRLG capacities 
 *   and edge disjointness requirements.
 * - It calculates potentials using the Shortest Path Faster Algorithm (SPFA) to determine 
 *   feasibility.
 * - If feasible, it constructs the paths while ensuring SRLG disjointness by avoiding 
 *   shared edges and cutting loops in the paths.
 * - The function supports both edge-disjoint and SRLG-disjoint path finding.
 * 
 * @note
 * - The function assumes that the graph is represented using dual and primal adjacency lists.
 * - The function uses several global variables, such as `Dual_Adj_List`, `SRLG_Capacities`, 
 *   `Primal_Adj_List`, and others, which must be properly initialized before calling this function.
 * - The function includes several assertions to ensure correctness during execution.
 * 
 * @warning
 * - The function assumes that the graph is well-formed and that the input parameters are valid.
 * - The function may enter an infinite loop or assert failure if the graph contains invalid data.
 */
bool find_srlg_disjoint_path(int Val, bool Print_it=false, bool find_proof=false) {

    assert(Val>0); // we are looking for at least one path

    // similar to Dual_Adj_List, but with different weights
    //vector<vector<pair<int, int> > > Adj_List(Dual_Node_Count+1);
    vector<vector<weighted_edge>> Adj_List(Dual_Node_Count+1);
    //iterate through each dual edge
    for (int Cur_Node=1; Cur_Node<=Dual_Node_Count; Cur_Node++) {
        for (auto x:Dual_Adj_List[Cur_Node]) {
            int Next_Node=x.dest, edge_weight=x.weight, id=x.edge_id;

            int additional_weight=0, bigger_node=max(Cur_Node, Next_Node); // ezt a 10 sort jo lenne atnezni
            if (bigger_node<=Primal_Area_Count) { // if both nodes are primal areas
                additional_weight=Val; // ez normál élnél a keresett utak száma lesz
            }
            if (bigger_node>Primal_Area_Count) { // if one of the nodes is a virtual node
                if (srlg_capacities[bigger_node]>=Val) { // elhagyjuk a túl nagy kapacitású régiókat
                    // additional_weight = Val; MÓDOSÍTÁS: a túl nagy kapacitású régiók helyett lecsökkentjük a kapacitásukat megfelelő méretűre
                    continue; 
                } else { // if capacity is less then the number of paths
                    additional_weight=srlg_capacities[bigger_node]; // virtuális csúcsnál pedig az SRLG kapacitása
                }
            }

            // calculate the edge weight
            int Next_Weight = edge_weight*Val + (id!=0 ? additional_weight : 0);
            Adj_List[Cur_Node].push_back({Next_Node, Next_Weight, id});
            
        }
    }
    // find the potential for the dual graph
    vector<int> Potential;

    Potential=find_potential_SPFA(Dual_Node_Count, Adj_List, find_proof);

    if (!Print_it) { // if we dont want to print, we can return
        return (Potential[0]!=-1);
    }

    if (Potential[0]==-1) { // If we have a negative cycle
        return false;
    }

    // if we have a feasible potential, then we can find the paths (only when we want to print too)
    Potential=find_potential_SPFA(Dual_Node_Count, Adj_List);
    final_potential=Potential;

    SRLG_Disjoint_Paths.clear();
    // if there is only one path, it is the special st path
    if (Val==1) {
        SRLG_Disjoint_Paths.push_back(Special_st_Path); // add the special st path to the solution
        all_paths.push_back(SRLG_Disjoint_Paths);
        all_srlg_capacities.push_back({}); // add empty capacities for virtual nodes
        for (int i = Primal_Area_Count+1; i <= Dual_Node_Count; i++) {
            all_srlg_capacities.back()[srlg_ids[i]] = srlg_capacities[i]; // add the capacities for virtual nodes
        }
        return true;
    }

    // iterate through the paths
    for (int path_id=0; path_id<Val; path_id++) {
        set<int> used_edges;
        vector<int> general_st_path, single_st_path; // general can contain loops. single is the final path

        int Prev_Node=Primal_Adj_List[Starting_Node][0], Cur_Node=Starting_Node; // itt prev_node nem is szamit

        while (Cur_Node!=Finishing_Node) {
            general_st_path.push_back(Cur_Node);

            assert(general_st_path.size()<=10*Node_Count); // why? - Peti: ha valamit elrontottam és körbe-körbe megy az út, akkor ezen elromlik
            int Next_Node=find_next_in_order(Prev_Node, Cur_Node);
            while (true) {
                int edge_id=Edge_Inverse[{Cur_Node, Next_Node}];
                if (used_edges.count(edge_id)) { // if edge is used, use the next available edge
                    Next_Node=find_next_in_order(Next_Node, Cur_Node);
                    continue;
                }
                
                int left_side=Potential[Primal_Edge_Left_Side[edge_id]], right_side=Potential[Primal_Edge_Right_Side[edge_id]];
                left_side+=1000*Val, right_side+=1000*Val; // why? - Peti:  a left side és right side is legyen pozitív (úgy, hogy a különbség megmarad), így normálisan működik a val-lal osztás
                int weight=Primal_Edge_Weight[edge_id];
                left_side+=weight*Val;
                
                // set left and right side based on the edge direction
                if (Primal_Edge_Left_Endpoint[edge_id]!=Cur_Node) {
                    assert(Primal_Edge_Right_Endpoint[edge_id]==Cur_Node);
                    swap(left_side, right_side);
                }


                int left_val=(left_side+path_id)/Val;
                int right_val=(right_side+path_id)/Val;
                if (left_val>right_val) {
                    assert(right_val+1==left_val); // why? - Peti: elvileg az élek súlya miatt két szomszédos terület potenciálja nem térhet el sokkal, talán segítene egy kisebb példa, hogy jobban megértsük.
                    used_edges.insert(edge_id);
                    break;
                }
                Next_Node=find_next_in_order(Next_Node, Cur_Node);
            }
            Prev_Node=Cur_Node;
            Cur_Node=Next_Node;
        }
        // add the finishing node to the path
        general_st_path.push_back(Finishing_Node);
        // for each node count how many times it is in the path (?)
        vector<int> Node_in_Path(Node_Count+1, 0);
        for (auto x:general_st_path) {
            Node_in_Path[x]++;
        }
        // cut the loops
        int pos=0;
        while (pos < general_st_path.size()) {
            int Cur_Node=general_st_path[pos];
            single_st_path.push_back(Cur_Node);
            while (Node_in_Path[Cur_Node]!=0) {
                Node_in_Path[general_st_path[pos]]--;
                pos++;
            }
        }
        // append to the SRLG disjoint paths
        SRLG_Disjoint_Paths.push_back(single_st_path);
    }
    all_srlg_capacities.push_back({}); // add empty capacities for virtual nodes
    all_paths.push_back(SRLG_Disjoint_Paths);
    for (int i = Primal_Area_Count+1; i <= Dual_Node_Count; i++) {
        all_srlg_capacities.back()[srlg_ids[i]] = srlg_capacities[i]; // add the capacities for virtual nodes
    }
    return true;
}

bool find_srlg_disjoint_walks(int Val, bool Print_it=false, bool find_proof=false) {
    cerr << "find_srlg_disjoint_walks started" << endl;
    assert(Val>0); // we are looking for at least one path

    // similar to Dual_Adj_List, but with different weights
    vector<vector<weighted_edge>> Adj_List(Dual_Node_Count+1);
    //iterate through each dual edge
    for (int Cur_Node=1; Cur_Node<=Dual_Node_Count; Cur_Node++) {
        for (auto x:Dual_Adj_List[Cur_Node]) {
            int Next_Node=x.dest, edge_weight=x.weight, id=x.edge_id;

            int additional_weight=0, bigger_node=max(Cur_Node, Next_Node); // ezt a 10 sort jo lenne atnezni
            if (bigger_node<=Primal_Area_Count) { // if both nodes are primal areas
                additional_weight=Val; // ez normál élnél a keresett utak száma lesz
            }
            if (bigger_node>Primal_Area_Count) { // if one of the nodes is a virtual node
                if (srlg_capacities[bigger_node]>=Val) { // elhagyjuk a túl nagy kapacitású régiókat
                    // additional_weight = Val; MÓDOSÍTÁS: a túl nagy kapacitású régiók helyett lecsökkentjük a kapacitásukat megfelelő méretűre
                    continue; 
                } else { // if capacity is less then the number of paths
                    additional_weight=srlg_capacities[bigger_node]; // virtuális csúcsnál pedig az SRLG kapacitása
                }
            }

            // calculate the edge weight
            int Next_Weight = edge_weight*Val + (id!=0 ? additional_weight : 0);
            Adj_List[Cur_Node].push_back({Next_Node, Next_Weight, id});
            
        }
    }
    // find the potential for the dual graph
    vector<int> Potential;

    Potential=find_potential_SPFA(Dual_Node_Count, Adj_List, find_proof);

    if (!Print_it) { // if we dont want to print, we can return
        return (Potential[0]!=-1);
    }

    if (Potential[0]==-1) { // If we have a negative cycle
        return false;
    }

    // if we have a feasible potential, then we can find the paths (only when we want to print too)
    Potential=find_potential_SPFA(Dual_Node_Count, Adj_List);
    final_potential=Potential;

    SRLG_Disjoint_Paths.clear();
    // if there is only one path, it is the special st path
    if (Val==1) {
        SRLG_Disjoint_Paths.push_back(Special_st_Path);
        return true;
    }

    // i-th element corresponds to path i
    next_edge_in_path.assign(Val, std::vector<int>(Edge_Count+1, 0));
    // we store each path's starting edges separately
    vector<int> starting_edges(Val, 0);
    cerr <<"start node: " << Starting_Node <<"\tend node: " << Finishing_Node << endl;
    for(int node=1; node<=Node_Count; node++)
    {
        cerr <<"node: " << Node_Id[node] <<endl;
        vector<vector<pair<int,int>>> multilevel_stack_in;
        vector<vector<pair<int,int>>> auxilary_for_out_edges;

        // iterate through the edges
        for (int next_node:Primal_Adj_List[node]) {
            int edge_id = Edge_Inverse[{node, next_node}];
            // get the potential
            int left_side=Potential[Primal_Edge_Left_Side[edge_id]], right_side=Potential[Primal_Edge_Right_Side[edge_id]];
            left_side+=1000*Val, right_side+=1000*Val;
            int weight=Primal_Edge_Weight[edge_id];
            left_side+=weight*Val;
            if (Primal_Edge_Left_Endpoint[edge_id]!=node) {
                assert(Primal_Edge_Right_Endpoint[edge_id]==node);
                swap(left_side, right_side);
            }
            // if the edge is an in edge
            if (left_side < right_side){
                
                cerr <<"\tedge: " << Primal_Edge_Id[edge_id] << endl;
                cerr << "\tleft node: " << Node_Id[Primal_Edge_Left_Endpoint[edge_id]] << "\t right node: " << Node_Id[Primal_Edge_Right_Endpoint[edge_id]]<<endl;
                cerr << "\tin edge for path:";
                vector<pair<int,int>> path_ids = {};
                // for each path id, check if it is a valid edge
                for (int path_id=0; path_id<Val; path_id++) {
                    int left_val=(left_side+path_id)/Val;
                    int right_val=(right_side+path_id)/Val;
                    if (left_val < right_val) { // in
                        cerr << " " << path_id;
                        path_ids.push_back({path_id, edge_id});
                    }
                }
                cerr << endl;
                assert(!path_ids.empty());
                multilevel_stack_in.push_back(path_ids);
            }

            // if the edge is an out edge
            if (left_side > right_side) { // out
                cerr <<"\tedge: " << Primal_Edge_Id[edge_id] << endl;
                cerr << "\tleft node: " << Node_Id[Primal_Edge_Left_Endpoint[edge_id]] << "\t right node: " << Node_Id[Primal_Edge_Right_Endpoint[edge_id]]<<endl;
                cerr << "\tout edge for path:";
                vector<pair<int,int>> path_ids = {};
                // for each path id, check if it is a valid edge
                for (int path_id=0; path_id<Val; path_id++) {
                    int left_val=(left_side+path_id)/Val;
                    int right_val=(right_side+path_id)/Val;
                    
                    if (left_val > right_val) { // out
                        cerr << " " << path_id;
                        path_ids.push_back({path_id, edge_id});
                    }
                }
                cerr << endl;
                // if there were out edges, add them to auxilary_for_out_edges
                if (!path_ids.empty()) {
                    auxilary_for_out_edges.push_back(path_ids);
                }
            }
        }
        if (node == Starting_Node) {
            cerr << "all the edges were scanned" <<endl;
            // we need to empty multilevel_stack_in for Starting_Node

            cerr <<"multilevel_stack_in start: " <<endl;
            for (vector<pair<int, int>> level: multilevel_stack_in)
            {
                for (pair<int,int> x: level)
                {
                    cerr << "<"<<x.first << ", " << x.second <<">, ";
                }
                cerr << endl;
            }
            cerr << "multilevel_stack_in end" <<endl;
            cerr <<"auxilary_for_out_edges start: " <<endl;
            for (vector<pair<int, int>> level: auxilary_for_out_edges)
            {
                for (pair<int,int> x: level)
                {
                    cerr << "<"<<x.first << ", " << x.second <<">, ";
                }
                cerr << endl;
            }
            cerr << "auxilary_for_out_edges end" <<endl;

            while(!multilevel_stack_in.empty()) {
                // remove the last element and store it
                vector<pair<int,int>> in_edges = multilevel_stack_in.back();
                multilevel_stack_in.pop_back();
                //iterate through edges
                for(auto in_edge: in_edges){
                    int path_id = in_edge.first;
                    int edge_id = in_edge.second;
                    cerr << "In edge: <" << path_id << ", " << edge_id << ">" << endl;

                    // find the first edge in auxilary_for_out_edges with the same path id
                    int level = 0;
                    while(level < auxilary_for_out_edges.size()){
                        vector<pair<int,int>>& out_edges = auxilary_for_out_edges[level];
                        auto it = std::find_if(out_edges.begin(), out_edges.end(),
                            [path_id](const std::pair<int, int>& p) {
                                return p.first == path_id;
                            });
                        if(it != out_edges.end()){ // if we found the edge
                            // set the edge pair in next_edge_in_path
                            cerr << "element found: <" << it->first << ", " << it->second << ">" << endl;
                            next_edge_in_path[path_id][edge_id] = it->second;
                            out_edges.erase(it);
                            if(out_edges.empty()){
                                auxilary_for_out_edges.erase(auxilary_for_out_edges.begin() + level);
                            }
                            level = auxilary_for_out_edges.size(); // break the loop
                        }
                        level += 1; //  go to the next level
                    }
                }
            }
            cerr << "No more in-edge" << endl;
            cerr << "starting edges: ";
            for(vector<pair<int,int>> level: auxilary_for_out_edges) {
                for(pair<int, int> edge: level) {
                    int path_id = edge.first;
                    int edge_id = edge.second;
                    cerr << "<" << path_id << ", " << Primal_Edge_Id[edge_id] << ">, ";
                    starting_edges[path_id] = edge_id;
                }
            }
            cerr << endl;
        }
        else { 
            cerr << "\tall the edges were scanned" <<endl;
            // we need to empty multilevel_stack_in for Starting_Node

            cerr <<"\t\tmultilevel_stack_in: " <<endl;
            for (vector<pair<int, int>> level: multilevel_stack_in)
            {
                cerr << "\t\t";
                for (pair<int,int> x: level)
                {
                    cerr << "<"<<x.first << ", " << Primal_Edge_Id[x.second] <<">, ";
                }
                cerr << endl;
            }
            cerr <<"\t\tauxilary_for_out_edges: " <<endl;
            for (vector<pair<int, int>> level: auxilary_for_out_edges)
            {
                cerr << "\t\t";
                for (pair<int,int> x: level)
                {
                    cerr << "<"<<x.first << ", " << Primal_Edge_Id[x.second] <<">, ";
                }
                cerr << endl;
            }
            // we need to empty auxilary_for_out_edges for Finishing_Node and other nodes
            while(!auxilary_for_out_edges.empty())
            {
                // remove the first element and store it
                vector<pair<int,int>> out_edges = auxilary_for_out_edges.front();
                auxilary_for_out_edges.erase(auxilary_for_out_edges.begin());
                // iterate through the edges
                for(auto out_edge: out_edges){
                    int path_id = out_edge.first;
                    int edge_id = out_edge.second;
                
                    // find the last edge in the multilevel_stack_in with the same path id
                    int level = multilevel_stack_in.size()-1;
                    while(level >= 0){
                        vector<pair<int,int>>& in_edges = multilevel_stack_in[level];
                        auto it = std::find_if(in_edges.begin(), in_edges.end(),
                            [path_id](const std::pair<int, int>& p) {
                                return p.first == path_id;
                            });
                        if(it != in_edges.end()) { // if we found the edge
                            // set the edge pair in next_edge_in_path
                            next_edge_in_path[path_id][it->second] = edge_id;
                            // remove the edge from the stack
                            in_edges.erase(it);
                            if(in_edges.empty()) {
                                multilevel_stack_in.erase(multilevel_stack_in.begin() + level); // remove the stack
                            }
                            level = -1; // break the loop
                        }
                        level -= 1; // go to the next level
                    }
                }
            }
        }    
    }
    // print paths for cerr as a temporary solution
    for (int edge = 1; edge <= Edge_Count; edge++){
        cerr << Primal_Edge_Id[edge] << " "; 
    }
    cerr << endl;
    for (int path_id=0; path_id<Val; path_id++) {
        cerr << "path " << path_id << ":";
        vector<int> path_edges = next_edge_in_path[path_id];
        for (int edge = 1; edge <= Edge_Count; edge++){

            cerr << " " << Primal_Edge_Id[path_edges[edge]];
        }
        cerr << endl;
    }
    
    // build walks one by one
    for (int path_id = 0; path_id < Val; path_id++) {
        // start with a walk containing only the starting node
        list<int> walk = {Starting_Node};
        vector<bool> seen_in_walk(Node_Count, false);
        seen_in_walk[Starting_Node] = true;

        // get current edge and endpoints
        int current_edge = starting_edges[path_id];
        int u = Primal_Edge_Left_Endpoint[current_edge];
        int v = Primal_Edge_Right_Endpoint[current_edge];

        
        // add the other node to the walk
        if (u == Starting_Node) swap(u, v);
        walk.push_back(u);
        seen_in_walk[u] = true;
        
        // get next_Edge and remove edge from the list
        int next_edge = next_edge_in_path[path_id][current_edge];
        next_edge_in_path[path_id][current_edge] = 0;
        // build the base walk
        while (next_edge != 0) {
            current_edge = next_edge;
            next_edge = next_edge_in_path[path_id][current_edge];
            next_edge_in_path[path_id][current_edge] = 0;
            u = Primal_Edge_Left_Endpoint[current_edge];
            v = Primal_Edge_Right_Endpoint[current_edge];
            if (walk.back() == u) swap(u, v);
            walk.push_back(u);
            seen_in_walk[u] = true;
        }

        // at this point, we are looking for cycles
        vector<list<int>> cycles = {};

        for (int edge_id: next_edge_in_path[path_id]) {
            if (edge_id != 0) // if there is a next edge
            {   
                vector<bool> seen_in_new_walk = seen_in_walk;
                int common_node = 0;
                // get current edge
                int current_edge = edge_id;
                int u = Primal_Edge_Left_Endpoint[current_edge];
                int v = Primal_Edge_Right_Endpoint[current_edge];

                // get next_Edge and remove edge from the list
                int next_edge = next_edge_in_path[path_id][current_edge];
                next_edge_in_path[path_id][current_edge] = 0;
                int x = Primal_Edge_Left_Endpoint[next_edge];
                int y = Primal_Edge_Right_Endpoint[next_edge];
                
                // build a temporary walk
                list<int> cycle = {};
                std::list<int>::iterator common_node_in_cycle;
                // u will be the last element of the cycle
                if (x == u || y == u) swap(u, v);
                cycle.push_back(v);
                seen_in_new_walk[v] = true;

                // if v is the common node, and it is the last node in the cycle
                if (common_node == 0 && seen_in_walk[v]) {
                    common_node = v;
                    common_node_in_cycle = std::prev(cycle.end());
                }
                
                // go through the rest of the cycle
                while (next_edge != 0) {
                    current_edge = next_edge;
                    // get next edge, and remove from the list
                    next_edge = next_edge_in_path[path_id][current_edge];
                    next_edge_in_path[path_id][current_edge] = 0;
                    u = Primal_Edge_Left_Endpoint[current_edge];
                    v = Primal_Edge_Right_Endpoint[current_edge];
                    // u should be the next node
                    if (u == cycle.back()) swap(u, v);
                    cycle.push_back(u);
                    seen_in_new_walk[u] = true;
                    // if u is the common node
                    if (common_node == 0 && seen_in_walk[u]) {
                        common_node = u;
                        common_node_in_cycle = std::prev(cycle.end());
                    }
                }

                // if we haven't found a common node, put this cycle aside
                if (common_node == 0) {
                    cycles.push_back(std::move(cycle));
                }
                else{
                    // Step 0: find the common node in the walk
                    auto common_node_in_walk = std::find(walk.begin(), walk.end(), common_node);

                    // Step 1: prepare a new list for the result
                    std::list<int> result;

                    // Step 2: move walk[:common_node_in_walk] into result
                    result.splice(result.end(), walk, walk.begin(), common_node_in_walk);

                    // Step 3: move cycle[common_node_in_cycle:] into result
                    result.splice(result.end(), cycle, common_node_in_cycle, cycle.end());

                    // Step 4: move cycle[:common_node_in_cycle] into result
                    result.splice(result.end(), cycle, next(cycle.begin()), cycle.end());

                    // Step 5: move walk[common_node_in_walk:] into result
                    result.splice(result.end(), walk, walk.begin(), walk.end());

                    // Step 6: move the rest of the walk into result
                    walk = std::move(result);
                    seen_in_walk = seen_in_new_walk;
                }
            }
        }

        // if there are cycles, we need to add them to the walk
        vector<list<int>>::iterator current_cycle = cycles.begin();

        // go through the cycles
        while (current_cycle != cycles.end()) {
            auto& cycle = *current_cycle;
            vector<bool> seen_in_new_walk = seen_in_walk; // reset the seen_in_new_walk

            // find the common node in the cycle
            list<int>::iterator common_node_in_cycle = cycle.end();
            for (auto it = cycle.begin(); it != cycle.end(); ++it) {
                if (common_node_in_cycle == cycle.end() && seen_in_walk[*it]) {
                    common_node_in_cycle = it;
                }
                seen_in_new_walk[*it] = true; // mark the node as seen in the new walk
            }
            
            if (common_node_in_cycle != cycle.end()) {
                // Step 0: find the common node in the walk
                auto common_node_in_walk = std::find(walk.begin(), walk.end(), *common_node_in_cycle);

                // Step 1: prepare a new list for the result
                std::list<int> result;

                // Step 2: move walk[:common_node_in_walk] into result
                result.splice(result.end(), walk, walk.begin(), common_node_in_walk);

                // Step 3: move cycle[common_node_in_cycle:] into result
                result.splice(result.end(), cycle, common_node_in_cycle, cycle.end());

                // Step 4: move cycle[:common_node_in_cycle] into result
                result.splice(result.end(), cycle, cycle.begin(), common_node_in_cycle);

                // Step 5: move walk[common_node_in_walk:] into result
                result.splice(result.end(), walk, common_node_in_walk, walk.end());

                // Step 6: move the rest of the walk into result
                walk = std::move(result);
                seen_in_walk = seen_in_new_walk;

                cycles.erase(current_cycle); // remove the cycle from the list
                current_cycle = cycles.begin(); // reset the iterator to the beginning
            } else {
                ++current_cycle; // go to the next cycle
            }
        }

        vector<int> path(walk.begin(), walk.end());
        SRLG_Disjoint_Paths.push_back(path);
        // print the walk
    }
    
    return true;
}

void find_area_boundaries() {
    for (int i=1; i<=Edge_Count; i++) {
        // get the left and right side of the edge
        int a=Primal_Edge_Left_Side[i], b=Primal_Edge_Right_Side[i];
        // add the edge to the primal areas
        Primal_Area_Boundary[a].insert(Primal_Edge_Left_Endpoint[i]);
        Primal_Area_Boundary[a].insert(Primal_Edge_Right_Endpoint[i]);
        Primal_Area_Boundary[b].insert(Primal_Edge_Left_Endpoint[i]);
        Primal_Area_Boundary[b].insert(Primal_Edge_Right_Endpoint[i]);
    }
}

void print_potential_and_neighbors() {
    cout << "potentials:" << endl;
    for (int i = 1; i <= Primal_Area_Count; i++) {
        // the first number in the row is the potential
        cout << final_potential[i];
        // then the rest of the row is the neighbors
        for (auto x:Primal_Area_Boundary[i]) {
            cout << " "<< Node_Id[x];
        }
        cout << endl;
    }
}

int find_min_cutting_srlg_index(bool edge_disjoint = false) {
    assert(!negative_cycle.empty());

    // Ha edge_disjoint == true, először próbáljunk csak többélű SRLG-kből választani
    if (edge_disjoint) {
        int min_index = -1;
        double min_val = numeric_limits<double>::max();

        for (int srlg_id : negative_cycle) {
            // Feltételezzük, hogy srlg_id indexelhető srlg_edges-ben
            if (srlg_edges[srlg_id].size() > 1) {  // csak azok, amik nem egyélűek
                int srlg_index = srlg_id_inverse[srlg_id];  // dual csúcs indexe
                double val = srlg_capacities[srlg_index] * srlg_probabilities[srlg_index];
                if (val < min_val) {
                    min_val = val;
                    min_index = srlg_index;
                }
            }
        }

        // Ha találtunk legalább egy többélű SRLG-t, azt adjuk vissza
        if (min_index != -1) {
            return min_index;
        }
        // ha nem találtunk, akkor esünk vissza az eredeti logikára (egyélűek is mehetnek)
    }

    // Eredeti működés: a negative_cycle összes SRLG-jéből választunk minimumot
    int min_index = srlg_id_inverse[negative_cycle[0]];
    double min_val = srlg_capacities[min_index] * srlg_probabilities[min_index];

    for (int srlg_id : negative_cycle) {
        int srlg_index = srlg_id_inverse[srlg_id];
        double val = srlg_capacities[srlg_index] * srlg_probabilities[srlg_index];
        if (val < min_val) {
            min_val = val;
            min_index = srlg_index;
        }
    }
    return min_index;
}


int find_min_srlg_index(int capacity_limit = 10)
{
    int min_index = -1;
    double min_val =  numeric_limits<double>::max();
    for(int i = Primal_Area_Count+1; i <= Dual_Node_Count; i++)
    {
        double current_val = srlg_capacities[i] * srlg_probabilities[i];
        if(srlg_capacities[i] < capacity_limit && current_val < min_val)
        {
            min_index = i;
            min_val = current_val;
        }
    }
    return min_index;
}


void print_all_solutions(int number_of_solutions = 10)
{
    for (int i = 0; i < number_of_solutions; i++) {
        cout << "number of paths: " << i+1 << endl;
        for (const auto& path : all_paths[i]) {
            for (int node : path) {
                cout << Node_Id[node] << " ";
            }
            cout << endl;
        }
        cout << "SRLG capacities: " << endl;
        for (const auto& pair : all_srlg_capacities[i]) {
            int srlg_id = pair.first-1;
            int capacity = pair.second;
            cout << srlg_id << " " << capacity << endl;
        }
    }
}


int main(int argc, char* argv[])
{
    string method = argv[1];
    int number_of_backups = stoi(argv[2]);
    int max_number_of_paths = stoi(argv[3]);
    bool edge_disjoint = false;
    if (argc > 4) {
        edge_disjoint = (string(argv[4]) == "edge_disjoint");
    }

    read_nodes();
    read_edges();
    check_primal_connectivity();
    order_neighbours();
    find_areas();
    Primal_Area_Boundary.resize(Primal_Area_Count+1);
    find_area_boundaries();

    find_special_st_path();
    process_the_st_path();

    read_SRLGs();
    add_dual_edges();

    // find initial solution
    int number_of_paths = 1;
    int iteration_counter = 1;
    find_srlg_disjoint_path(1, true);

    // linear search for the number of paths
    while(find_srlg_disjoint_path(number_of_paths+1, false)) {
        find_srlg_disjoint_path(number_of_paths+1, true);
        number_of_paths++;
    }

    while (number_of_paths < max_number_of_paths)
    {
        cerr << iteration_counter << endl;
        bool found_new_path = find_srlg_disjoint_path(number_of_paths+1, false, true); //find negative cycle but dont modify solution
        if (!found_new_path) {

            // find the SRLG index to improve based on the method
            int srlg_index_to_improve = -1;
            if (method == "min_cutting_prob") srlg_index_to_improve = find_min_cutting_srlg_index(edge_disjoint);
            else if (method == "min_prob") srlg_index_to_improve = find_min_srlg_index();

            // improve the SRLG index
            srlg_capacities[srlg_index_to_improve]++;
            // if there are backups, increase it as much as possible.
            // if we can only correct number_of_backups errors, then if an srlg's capacity is bigger, it can be set to infinity
            if (number_of_backups > 0 && srlg_capacities[srlg_index_to_improve] > number_of_backups) {
                srlg_capacities[srlg_index_to_improve] = max_number_of_paths;
            }
        }
        else {
            number_of_paths++;
            find_srlg_disjoint_path(number_of_paths, true); // modify solution but do not find negative cycle
        }
    }

    print_all_solutions(number_of_paths);
    return 0;
}