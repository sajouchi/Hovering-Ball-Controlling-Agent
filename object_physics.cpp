# include <iostream>
# include <cmath>
# include <cstdlib>  // <--- ADD THIS for random numbers
# include <ctime>    // <--- ADD THIS for time seeding
using namespace std;

class rl_env
{
    private:
        double gravity = -9.8;
        double dt = 0.01;
        double target_h = 10.0;
        double max_height = 20.0;
        double max_steps = 500;
        double force = 15.0;

        double h;
        double v;
        int steps_count;

    public:
        bool done = false; // to see if the limits gets broken
        bool terminal = true; // to monitor the step reaching the decided run limit/timeout for the max_steps and to perform flagging

        void reset()
        {
            h = 1.0 + (rand()%180)/10.0; // default starting position/fixed for now for simplicity
            // v = ((rand()%100)/10.0) - 5.0;
            v=0; // for simplicity

            steps_count = 0;
            done = false;
            terminal=true;

            cout<<h<<" "<<v<<" "<<"0"<<" "<<done<<" "<<terminal<<endl<<flush;
        }

        void step(int action)
        {
            double action_force = 0.0;
            if(action == 0)
            {
                action_force -= force;
            }
            else if(action == 2)
            {
                action_force += force;
            }

            double acceleration = gravity + action_force;
            v = v + acceleration * dt;
            v = max(min(v, 10.0), -10.0);
            // if(v>10.0){v=10.0;}
            // else if(v<-10.0){v=-10.0;} // did to keep the velocity in check under limits


            h = h + v * dt;

            // double reward = -abs(h - target_h); // old reward formula resulted in greedy quick crash by the agent. The quick crash was getting less penalty then staying at wrong postion near the targets , so agents wuld choose to crash immediatly around 10 steps
            // double reward = 1.0 - (abs(h-target_h)/10.0); // will reward per step. Tried for positive rewarding for trying to hover in correct areas
            double dist_error = abs(h-target_h);
            double rew_dist = exp(-2.0 * dist_error);
            double rew_val = -0.1*abs(v);
            double rew_alive = 0.05;

            double reward = rew_dist + rew_val + rew_alive;

            steps_count++;

            if(h<=0 || h > max_height)
            {
                reward = -2.0;
                done = true;
                terminal=true;
            }
            else if(steps_count >= max_steps)
            {
                done = true;
                terminal=false;
            }

            cout << h <<" "<<v<<" "<<reward<<" "<<done<<" "<<terminal<<endl<<flush;
        }

        void get_state(double &pos, double &val)
        {
            pos = h;
            val = v;
        }
};

int main()
{
    srand(time(0)); // Seed the random generator

    rl_env env;
    env.reset();

    while(!env.done)
    {
        int action;
        cin>>action;
        env.step(action);
    }
    return 0;
}