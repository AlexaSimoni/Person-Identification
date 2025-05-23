

Before running the project you have to install the following:
download mongoDB and set in your c:/.../Program Files :
https://www.mongodb.com/try/download/community

download FlowNetPytorch project and set it into FlowNet_Component directory:
https://github.com/ClementPinard/FlowNetPytorch.git

download flownets_from_caffe.pth and set it into FlowNet_Component/checkpoints directory:
https://drive.google.com/drive/folders/16eo3p9dO_vmssxRoZCmWkTpNjKRzJzn5

Requirements:

Python 3.10
FastAPI
Uvicorn
OpenCV
Requests
Motor (for MongoDB)
facenet-pytorch
PIL
numpy

Then to run the project:

Clone the Repository:
git clone https://github.com/your-username/your-repo.git
cd your-repo

Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install Dependencies:
pip install -r server/requirements.txt


Configure MongoDB:
Ensure you have MongoDB running locally or provide a connection URL in the .env file:
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://root:example@localhost:27017/?authMechanism=DEFAULT")
If installed locally then run:
mongod 


Run the Application:
uvicorn server.main:app --reload      

Open second Terminal to run client:

cd client
npm install
npm start  




