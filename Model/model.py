{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras import layers\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import models\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from PIL import Image as img\n",
    "from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten\n",
    "from keras.callbacks import EarlyStopping\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "from tensorflow.keras.layers.experimental import preprocessing\n",
    "import cv2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv('./data/HAM10000_metadata.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['path'] = './data/HAM10000/' + data['image_id'] + '.jpg'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>lesion_id</th>\n",
       "      <th>image_id</th>\n",
       "      <th>dx</th>\n",
       "      <th>dx_type</th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>localization</th>\n",
       "      <th>path</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>HAM_0000118</td>\n",
       "      <td>ISIC_0027419</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "      <td>./data/HAM10000/ISIC_0027419.jpg</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>HAM_0000118</td>\n",
       "      <td>ISIC_0025030</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "      <td>./data/HAM10000/ISIC_0025030.jpg</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>HAM_0002730</td>\n",
       "      <td>ISIC_0026769</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "      <td>./data/HAM10000/ISIC_0026769.jpg</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>HAM_0002730</td>\n",
       "      <td>ISIC_0025661</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "      <td>./data/HAM10000/ISIC_0025661.jpg</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>HAM_0001466</td>\n",
       "      <td>ISIC_0031633</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>75.0</td>\n",
       "      <td>male</td>\n",
       "      <td>ear</td>\n",
       "      <td>./data/HAM10000/ISIC_0031633.jpg</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     lesion_id      image_id   dx dx_type   age   sex localization  \\\n",
       "0  HAM_0000118  ISIC_0027419  bkl   histo  80.0  male        scalp   \n",
       "1  HAM_0000118  ISIC_0025030  bkl   histo  80.0  male        scalp   \n",
       "2  HAM_0002730  ISIC_0026769  bkl   histo  80.0  male        scalp   \n",
       "3  HAM_0002730  ISIC_0025661  bkl   histo  80.0  male        scalp   \n",
       "4  HAM_0001466  ISIC_0031633  bkl   histo  75.0  male          ear   \n",
       "\n",
       "                               path  \n",
       "0  ./data/HAM10000/ISIC_0027419.jpg  \n",
       "1  ./data/HAM10000/ISIC_0025030.jpg  \n",
       "2  ./data/HAM10000/ISIC_0026769.jpg  \n",
       "3  ./data/HAM10000/ISIC_0025661.jpg  \n",
       "4  ./data/HAM10000/ISIC_0031633.jpg  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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RESSx0BEREREksdARERERJLHQERERESSx0BEREREksdARERERJLHQERERESSx0BEREREksdARERERJLHQERERESSx0BEREREksdARERERJLHQERERESSx0BEREREksdARERERJLHQERERESSx0BEREREksdARERERJLHQERERESSx0BEREREksdARERERJLHQERERESSx0BEREREksdARERERJLHQERERESSx0BEREREksdARERERJLHQERERESSx0BEREREksdARERERJLHQERERESSx0BEREREkmfQQDRlyhTIZDKtV+3atcXlOTk5CA0Nhb29PaysrBAUFIT09HStMZKTkxEYGAhLS0s4Ojpi3LhxyM/P1+qzb98+NGrUCHK5HDVq1EB0dHRZbB4RERG9IQx+hKhOnTpITU0VX//884+4bMyYMfj777+xadMm7N+/H7du3UKPHj3E5QUFBQgMDEReXh4OHz6M1atXIzo6GpMnTxb7XL16FYGBgWjdujUSExMRHh6OoUOHYteuXWW6nURERFR+mRh8AiYmcHZ2LtL+8OFDrFq1CuvWrUObNm0AAFFRUfDy8sKRI0fQrFkzxMTE4Pz589i9ezecnJzQsGFDTJ8+HRMmTMCUKVNgZmaG5cuXw9PTE5GRkQAALy8v/PPPP5g/fz4CAgLKdFuJiIiofDJ4ILp48SJcXV1hbm4OX19fzJo1C+7u7oiPj4darUa7du3EvrVr14a7uzvi4uLQrFkzxMXFoV69enBychL7BAQEYMSIETh37hzeffddxMXFaY1R2Cc8PPyFc8rNzUVubq74XqVSAQDUajXUanUJbTnEMZ/+L72YRqMBAMhNZBCMBb3WlZnIYGFhAY1GI4lac7/SHWulH9ZLd6yV7kqrVvqMZ9BA1LRpU0RHR6NWrVpITU3F1KlT0bJlS5w9exZpaWkwMzODra2t1jpOTk5IS0sDAKSlpWmFocLlhcte1kelUiE7OxsWFhZF5jVr1ixMnTq1SHtMTAwsLS2Lvb0vo1QqS2Xciujbju4ACvRcywPosh4pKSlISUkpjWmVS9yvdMda6Yf10h1rpbuSrlVWVpbOfQ0aiDp27Cj+f/369dG0aVN4eHhg48aNzw0qZWXixImIiIgQ36tUKri5ucHf3x8KhaJEP0utVkOpVKJ9+/YwNTUt0bErmoSEBKSmpmLC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YmNjS2xyRERERGWhWEeIDh48iMOHD8PMzEyrvVq1akhJSSmRiRERERGVlWIdIdJoNCgoKCjSfvPmTVhbW7/2pIiIiIjKUrECkb+/PxYsWCC+l8lkyMzMxNdff82v8yAiIqI3TrFOmUVGRiIgIADe3t7IyclB3759cfHiRTg4OGD9+vUlPUciIiKiUlWsQFS1alWcOnUKv/76K06fPo3MzEyEhISgX79+WhdZExEREb0JihWIAMDExAT9+/cvybkQERERGUSxAtGaNWteunzgwIHFmgwRERGRIRQrEI0ePVrrvVqtRlZWFszMzGBpaclARERERG+UYt1l9uDBA61XZmYmkpKS0KJFC15UTURERG+cYn+X2bNq1qyJ2bNnFzl6RERERFTelVggAp5caH3r1q2SHJKIiIio1BXrGqK//vpL670gCEhNTcX333+P999/v0QmRkRERFRWihWIunXrpvVeJpOhSpUqaNOmDSIjI0tiXkRERERlpliBSKPRlPQ8iIiIiAymRK8hIiIiInoTFesIUUREhM59582bV5yPICIiIiozxQpECQkJSEhIgFqtRq1atQAA//33H4yNjdGoUSOxn0wmK5lZEhEREZWiYgWiLl26wNraGqtXr0blypUBPHlY4+DBg9GyZUt89tlnJTpJIiIiotJUrGuIIiMjMWvWLDEMAUDlypUxY8YM3mVGREREb5xiBSKVSoU7d+4Uab9z5w4ePXr02pMiIiIiKkvFCkTdu3fH4MGDsXnzZty8eRM3b97E77//jpCQEPTo0aOk50hERERUqop1DdHy5csxduxY9O3bF2q1+slAJiYICQnB3LlzS3SCRERERKWtWIHI0tISS5cuxdy5c3H58mUAQPXq1VGpUqUSnRwRERFRWXitBzOmpqYiNTUVNWvWRKVKlSAIQknNi4iIiKjMFCsQ3bt3D23btsU777yDTp06ITU1FQAQEhLCW+6JiIjojVOsQDRmzBiYmpoiOTkZlpaWYnuvXr2wc+fOEpscERERUVko1jVEMTEx2LVrF6pWrarVXrNmTVy/fr1EJkZERERUVop1hOjx48daR4YK3b9/H3K5/LUnRURERFSWihWIWrZsiTVr1ojvZTIZNBoN5syZg9atW+s8zrJly1C/fn0oFAooFAr4+vpix44d4vKcnByEhobC3t4eVlZWCAoKQnp6utYYycnJCAwMhKWlJRwdHTFu3Djk5+dr9dm3bx8aNWoEuVyOGjVqIDo6ujibTURERBVUsU6ZzZkzB23btsWJEyeQl5eH8ePH49y5c7h//z4OHTqk8zhVq1bF7NmzUbNmTQiCgNWrV6Nr165ISEhAnTp1MGbMGGzbtg2bNm2CjY0NRo4ciR49eoifUVBQgMDAQDg7O+Pw4cNITU3FwIEDYWpqipkzZwIArl69isDAQAwfPhxr165FbGwshg4dChcXFwQEBBRn84mIiKiCKVYgqlu3Lv777z98//33sLa2RmZmJnr06IHQ0FC4uLjoPE6XLl203n/zzTdYtmwZjhw5gqpVq2LVqlVYt24d2rRpAwCIioqCl5cXjhw5gmbNmiEmJgbnz5/H7t274eTkhIYNG2L69OmYMGECpkyZAjMzMyxfvhyenp7id6x5eXnhn3/+wfz58xmIiIiICEAxApFarUaHDh2wfPlyfPnllyU2kYKCAmzatAmPHz+Gr68v4uPjoVar0a5dO7FP7dq14e7ujri4ODRr1gxxcXGoV68enJycxD4BAQEYMWIEzp07h3fffRdxcXFaYxT2CQ8Pf+FccnNzkZubK75XqVQAnmx74ZO5S0rheCU9bkWk0WgAAHITGQRj/Z55JTORwcLCAhqNRhK15n6lO9ZKP6yX7lgr3ZVWrfQZT+9AZGpqitOnT+u72gudOXMGvr6+yMnJgZWVFf744w94e3sjMTERZmZmsLW11erv5OSEtLQ0AEBaWppWGCpcXrjsZX1UKhWys7NhYWFRZE6zZs3C1KlTi7THxMQ892LykqBUKktl3Iro247uAAr0XMsD6LIeKSkpSElJKY1plUvcr3THWumH9dIda6W7kq5VVlaWzn2Ldcqsf//+WLVqFWbPnl2c1bXUqlULiYmJePjwIX777TcEBwdj//79rz3u65g4cSIiIiLE9yqVCm5ubvD394dCoSjRz1Kr1VAqlWjfvj1MTU1LdOyKJiEhAampqZiwIxmCvade6+alX0H6us9x4MABNGjQoJRmWH5wv9Ida6Uf1kt3rJXuSqtWhWd4dFGsQJSfn4+ffvoJu3fvho+PT5HvMJs3b57OY5mZmaFGjRoAAB8fHxw/fhwLFy5Er169kJeXh4yMDK2jROnp6XB2dgYAODs749ixY1rjFd6F9nSfZ+9MS09Ph0KheO7RIQCQy+XPfXyAqalpqe3UpTl2RWFk9OSmyNx8AUKBTK91c/MFZGdnw8jISFJ15n6lO9ZKP6yX7lgr3ZV0rfQZS6/b7q9cuQKNRoOzZ8+iUaNGsLa2xn///YeEhATxlZiYqO98tWg0GuTm5sLHxwempqaIjY0VlyUlJSE5ORm+vr4AAF9fX5w5cwa3b98W+yiVSigUCnh7e4t9nh6jsE/hGERERER6HSGqWbMmUlNTsXfvXgBPvqpj0aJFRa7R0dXEiRPRsWNHuLu749GjR1i3bh327duHXbt2wcbGBiEhIYiIiICdnR0UCgXCwsLg6+uLZs2aAQD8/f3h7e2NAQMGYM6cOUhLS8OkSZMQGhoqHuEZPnw4vv/+e4wfPx5DhgzBnj17sHHjRmzbtq1YcyYiIqKKR69A9Oy32e/YsQOPHz8u9offvn0bAwcORGpqKmxsbFC/fn3s2rUL7du3BwDMnz8fRkZGCAoKQm5uLgICArB06VJxfWNjY2zduhUjRoyAr68vKlWqhODgYEybNk3s4+npiW3btmHMmDFYuHAhqlatipUrV/KWeyIiIhIV6xqiQs8GJH2tWrXqpcvNzc2xZMkSLFmy5IV9PDw8sH379peO4+fnh4SEhGLNkYiIiCo+va4hkslkkMlkRdqIiIiI3mR6nzIbNGiQeH1OTk4Ohg8fXuQus82bN5fcDImIiIhKmV6BKDg4WOt9//79S3QyRERERIagVyCKiooqrXkQERERGYxe1xARERERVUQMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkGDUSzZs3Ce++9B2trazg6OqJbt25ISkrS6pOTk4PQ0FDY29vDysoKQUFBSE9P1+qTnJyMwMBAWFpawtHREePGjUN+fr5Wn3379qFRo0aQy+WoUaMGoqOjS3vziIiI6A1h0EC0f/9+hIaG4siRI1AqlVCr1fD398fjx4/FPmPGjMHff/+NTZs2Yf/+/bh16xZ69OghLi8oKEBgYCDy8vJw+PBhrF69GtHR0Zg8ebLY5+rVqwgMDETr1q2RmJiI8PBwDB06FLt27SrT7SUiIqLyycSQH75z506t99HR0XB0dER8fDxatWqFhw8fYtWqVVi3bh3atGkDAIiKioKXlxeOHDmCZs2aISYmBufPn8fu3bvh5OSEhg0bYvr06ZgwYQKmTJkCMzMzLF++HJ6enoiMjAQAeHl54Z9//sH8+fMREBBQ5ttNRERE5Uu5uobo4cOHAAA7OzsAQHx8PNRqNdq1ayf2qV27Ntzd3REXFwcAiIuLQ7169eDk5CT2CQgIgEqlwrlz58Q+T49R2KdwDCIiIpI2gx4heppGo0F4eDjef/991K1bFwCQlpYGMzMz2NraavV1cnJCWlqa2OfpMFS4vHDZy/qoVCpkZ2fDwsJCa1lubi5yc3PF9yqVCgCgVquhVqtfc0u1FY5X0uNWRBqNBgAgN5FBMBb0WldmIoOFhQU0Go0kas39SneslX5YL92xVrorrVrpM165CUShoaE4e/Ys/vnnH0NPBbNmzcLUqVOLtMfExMDS0rJUPlOpVJbKuBXRtx3dARTouZYH0GU9UlJSkJKSUhrTKpe4X+mOtdIP66U71kp3JV2rrKwsnfuWi0A0cuRIbN26FQcOHEDVqlXFdmdnZ+Tl5SEjI0PrKFF6ejqcnZ3FPseOHdMar/AutKf7PHtnWnp6OhQKRZGjQwAwceJEREREiO9VKhXc3Nzg7+8PhULxehv7DLVaDaVSifbt28PU1LREx65oEhISkJqaigk7kiHYe+q1bl76FaSv+xwHDhxAgwYNSmmG5Qf3K92xVvphvXTHWumutGpVeIZHFwYNRIIgICwsDH/88Qf27dsHT0/tf+R8fHxgamqK2NhYBAUFAQCSkpKQnJwMX19fAICvry+++eYb3L59G46OjgCeJEyFQgFvb2+xz/bt27XGViqV4hjPksvlkMvlRdpNTU1LbacuzbErCiOjJ5e85eYLEApkeq2bmy8gOzsbRkZGkqoz9yvdsVb6Yb10x1rprqRrpc9YBg1EoaGhWLduHbZs2QJra2vxmh8bGxtYWFjAxsYGISEhiIiIgJ2dHRQKBcLCwuDr64tmzZoBAPz9/eHt7Y0BAwZgzpw5SEtLw6RJkxAaGiqGmuHDh+P777/H+PHjMWTIEOzZswcbN27Etm3bDLbtREREVH4Y9C6zZcuW4eHDh/Dz84OLi4v42rBhg9hn/vz56Ny5M4KCgtCqVSs4Oztj8+bN4nJjY2Ns3boVxsbG8PX1Rf/+/TFw4EBMmzZN7OPp6Ylt27ZBqVSiQYMGiIyMxMqVK3nLPREREQEoB6fMXsXc3BxLlizBkiVLXtjHw8OjyCmxZ/n5+SEhIUHvORIREVHFV66eQ0RERERkCAxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5DEREREQkeQxEREREJHkMRERERCR5JoaeABFReXDq1CkYGen/O6KDgwPc3d1LYUZEVJYYiMoJ/mVMZBg3b94EALRq1QrZ2dl6r29uYYmkfy/w55DoDcdAZGD8y5jIsO7duwcAsOsQhgKFq17rqu/dwL2tkbh79y5/BonecAYNRAcOHMDcuXMRHx+P1NRU/PHHH+jWrZu4XBAEfP311/jxxx+RkZGB999/H8uWLUPNmjXFPvfv30dYWBj+/vtvGBkZISgoCAsXLoSVlZXY5/Tp0wgNDcXx48dRpUoVhIWFYfz48WW5qS/Ev4yJygdTu7dg4lDd0NMgIgMxaCB6/PgxGjRogCFDhqBHjx5Fls+ZMweLFi3C6tWr4enpia+++goBAQE4f/48zM3NAQD9+vVDamoqlEol1Go1Bg8ejGHDhmHdunUAAJVKBX9/f7Rr1w7Lly/HmTNnMGTIENja2mLYsGFlur0vw7+MiYiIDMeggahjx47o2LHjc5cJgoAFCxZg0qRJ6Nq1KwBgzZo1cHJywp9//onevXvjwoUL2LlzJ44fP47GjRsDABYvXoxO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ikJycjL27dtXAltERIbCQEREb6zatWujefPm+OmnnwAAly5dwsGDBxESEoKCggJMnz4d9erVg52dHaysrLBr1y4kJycX+/NOnTqFS5cuwdraGlZWVrCysoKdnR1ycnJw+fLlktosIjIAE0NPgIjodYSEhCAsLAxLlixBVFQUqlevjg8++ADffvstFi5ciAULFqBevXqoVKkSwsPDX3pqSyaTaZ1+A56cJiuUmZkJHx8frF27tsi6VapUKbmNIqIyx0BERG+0jz/+GKNHj8a6deuwZs0ajBgxAjKZDIcOHULXrl3Rv39/AIBGo8F///0Hb2/vF45VpUoVpKamiu8vXryIrKws8X2jRo2wYcMGODo6QqFQlN5GEVGZ4ykzInqjWVlZoVevXpg4cSJSU1MxaNAgAEDNmjWhVCpx+PBhXLhwAZ9++inS09NfOlabNm3w/fffIyEhASdOnMDw4cNhamoqLu/Xrx8cHBzQtWtXHDx4EFevXsW+ffswatSo597+T0RvDgYiInrjhYSE4MGDBwgICICrqysAYNKkSWjUqBECAgLg5+cHZ2dndOvW7aXjREZGws3NDS1btkTfvn0xduxYWFpaisstLS1x4MABuLu7o0ePHvDy8kJISAhycnJ4xIjoDScTnj1hTkRERCQxPEJEREREksdARERERJLHQERERESSx0BEREREksdARERERJLHQERERESSx0BEREREksdARERERJLHQERERESSx0BEREREksdARERERJLHQERERESS9/8A7pj/NvpKcZwAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.hist(data['dx'], bins=30, edgecolor='black')\n",
    "plt.title('Histogram of Random Data')\n",
    "plt.xlabel('Value')\n",
    "plt.ylabel('Frequency')\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "one_hot = pd.get_dummies(data['dx'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>akiec</th>\n",
       "      <th>bcc</th>\n",
       "      <th>bkl</th>\n",
       "      <th>df</th>\n",
       "      <th>mel</th>\n",
       "      <th>nv</th>\n",
       "      <th>vasc</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   akiec  bcc  bkl  df  mel  nv  vasc\n",
       "0      0    0    1   0    0   0     0\n",
       "1      0    0    1   0    0   0     0\n",
       "2      0    0    1   0    0   0     0\n",
       "3      0    0    1   0    0   0     0\n",
       "4      0    0    1   0    0   0     0"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "one_hot.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.concat([data, one_hot], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>lesion_id</th>\n",
       "      <th>image_id</th>\n",
       "      <th>dx</th>\n",
       "      <th>dx_type</th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>localization</th>\n",
       "      <th>path</th>\n",
       "      <th>akiec</th>\n",
       "      <th>bcc</th>\n",
       "      <th>bkl</th>\n",
       "      <th>df</th>\n",
       "      <th>mel</th>\n",
       "      <th>nv</th>\n",
       "      <th>vasc</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>HAM_0000118</td>\n",
       "      <td>ISIC_0027419</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "      <td>./data/HAM10000/ISIC_0027419.jpg</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>HAM_0000118</td>\n",
       "      <td>ISIC_0025030</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "      <td>./data/HAM10000/ISIC_0025030.jpg</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>HAM_0002730</td>\n",
       "      <td>ISIC_0026769</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "      <td>./data/HAM10000/ISIC_0026769.jpg</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>HAM_0002730</td>\n",
       "      <td>ISIC_0025661</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "      <td>./data/HAM10000/ISIC_0025661.jpg</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>HAM_0001466</td>\n",
       "      <td>ISIC_0031633</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>75.0</td>\n",
       "      <td>male</td>\n",
       "      <td>ear</td>\n",
       "      <td>./data/HAM10000/ISIC_0031633.jpg</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10010</th>\n",
       "      <td>HAM_0002867</td>\n",
       "      <td>ISIC_0033084</td>\n",
       "      <td>akiec</td>\n",
       "      <td>histo</td>\n",
       "      <td>40.0</td>\n",
       "      <td>male</td>\n",
       "      <td>abdomen</td>\n",
       "      <td>./data/HAM10000/ISIC_0033084.jpg</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10011</th>\n",
       "      <td>HAM_0002867</td>\n",
       "      <td>ISIC_0033550</td>\n",
       "      <td>akiec</td>\n",
       "      <td>histo</td>\n",
       "      <td>40.0</td>\n",
       "      <td>male</td>\n",
       "      <td>abdomen</td>\n",
       "      <td>./data/HAM10000/ISIC_0033550.jpg</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10012</th>\n",
       "      <td>HAM_0002867</td>\n",
       "      <td>ISIC_0033536</td>\n",
       "      <td>akiec</td>\n",
       "      <td>histo</td>\n",
       "      <td>40.0</td>\n",
       "      <td>male</td>\n",
       "      <td>abdomen</td>\n",
       "      <td>./data/HAM10000/ISIC_0033536.jpg</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10013</th>\n",
       "      <td>HAM_0000239</td>\n",
       "      <td>ISIC_0032854</td>\n",
       "      <td>akiec</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>face</td>\n",
       "      <td>./data/HAM10000/ISIC_0032854.jpg</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10014</th>\n",
       "      <td>HAM_0003521</td>\n",
       "      <td>ISIC_0032258</td>\n",
       "      <td>mel</td>\n",
       "      <td>histo</td>\n",
       "      <td>70.0</td>\n",
       "      <td>female</td>\n",
       "      <td>back</td>\n",
       "      <td>./data/HAM10000/ISIC_0032258.jpg</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>10015 rows × 15 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "         lesion_id      image_id     dx dx_type   age     sex localization  \\\n",
       "0      HAM_0000118  ISIC_0027419    bkl   histo  80.0    male        scalp   \n",
       "1      HAM_0000118  ISIC_0025030    bkl   histo  80.0    male        scalp   \n",
       "2      HAM_0002730  ISIC_0026769    bkl   histo  80.0    male        scalp   \n",
       "3      HAM_0002730  ISIC_0025661    bkl   histo  80.0    male        scalp   \n",
       "4      HAM_0001466  ISIC_0031633    bkl   histo  75.0    male          ear   \n",
       "...            ...           ...    ...     ...   ...     ...          ...   \n",
       "10010  HAM_0002867  ISIC_0033084  akiec   histo  40.0    male      abdomen   \n",
       "10011  HAM_0002867  ISIC_0033550  akiec   histo  40.0    male      abdomen   \n",
       "10012  HAM_0002867  ISIC_0033536  akiec   histo  40.0    male      abdomen   \n",
       "10013  HAM_0000239  ISIC_0032854  akiec   histo  80.0    male         face   \n",
       "10014  HAM_0003521  ISIC_0032258    mel   histo  70.0  female         back   \n",
       "\n",
       "                                   path  akiec  bcc  bkl  df  mel  nv  vasc  \n",
       "0      ./data/HAM10000/ISIC_0027419.jpg      0    0    1   0    0   0     0  \n",
       "1      ./data/HAM10000/ISIC_0025030.jpg      0    0    1   0    0   0     0  \n",
       "2      ./data/HAM10000/ISIC_0026769.jpg      0    0    1   0    0   0     0  \n",
       "3      ./data/HAM10000/ISIC_0025661.jpg      0    0    1   0    0   0     0  \n",
       "4      ./data/HAM10000/ISIC_0031633.jpg      0    0    1   0    0   0     0  \n",
       "...                                 ...    ...  ...  ...  ..  ...  ..   ...  \n",
       "10010  ./data/HAM10000/ISIC_0033084.jpg      1    0    0   0    0   0     0  \n",
       "10011  ./data/HAM10000/ISIC_0033550.jpg      1    0    0   0    0   0     0  \n",
       "10012  ./data/HAM10000/ISIC_0033536.jpg      1    0    0   0    0   0     0  \n",
       "10013  ./data/HAM10000/ISIC_0032854.jpg      1    0    0   0    0   0     0  \n",
       "10014  ./data/HAM10000/ISIC_0032258.jpg      0    0    0   0    1   0     0  \n",
       "\n",
       "[10015 rows x 15 columns]"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_image(image_path):\n",
    "    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)    \n",
    "    if image is not None:\n",
    "        image = cv2.resize(image, (28,28))\n",
    "        image = image.astype(np.float32) / 255.0\n",
    "        image = np.expand_dims(image, axis=-1)\n",
    "    return image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "images = [load_image(image_path) for image_path in data['path']]\n",
    "images = np.array(images)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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PXu2KioqdODAAX37299WY2Ojnn76aYerHW3CFxD+ora2duTfS5cuVXV1tWbPnq0nn3xSX/va1xyuDBPBTTfdNPLvJUuWaOnSpZo3b5527typ1atXO1zZ+Kirq9PBgwc/Fc+DfpyzHYfbbrtt5N9LlixReXm5Vq9erebmZs2bN+98L/OMJvyv4IqKipSenv6RV7G0t7errKzM0aomhvz8fF1wwQVqampyvRRn3j8HOD8+au7cuSoqKpqS58fGjRv1/PPP6ze/+c2ot28pKyvT0NCQurq6Rm0/Vc+Hsx2HM6murpakCXU+TPgCCoVCWr58ubZv3z7yuWQyqe3bt2vlypUOV+ZeX1+fmpubVV5e7nopzlRVVamsrGzU+dHT06NXXnnlU39+tLS0qLOzc0qdH57naePGjXrmmWe0Y8cOVVVVjbp++fLlCgaDo86HxsZGHTlyZEqdD+c6Dmeyf/9+SZpY54PrV0Gk4oknnvDC4bC3detW74033vBuu+02Lz8/32tra3O9tPPqb//2b72dO3d6hw8f9l5++WWvpqbGKyoq8o4fP+56aeOqt7fXe+2117zXXnvNk+T98Ic/9F577TXv3Xff9TzP877//e97+fn53nPPPecdOHDAu/baa72qqipvYGDA8crH1scdh97eXu+ee+7xdu/e7R0+fNh78cUXvc9+9rPeggULvMHBQddLHzN33HGHl5eX5+3cudNrbW0duUSj0ZFtbr/9dm/WrFnejh07vL1793orV670Vq5c6XDVY+9cx6Gpqcn7+7//e2/v3r3e4cOHveeee86bO3eut2rVKscrH21SFJDned5PfvITb9asWV4oFPJWrFjh7dmzx/WSzrsbb7zRKy8v90KhkDdjxgzvxhtv9Jqamlwva9z95je/8SR95LJhwwbP806/FPu73/2uV1pa6oXDYW/16tVeY2Oj20WPg487DtFo1FuzZo1XXFzsBYNBb/bs2d6tt9465X5IO9PXL8l75JFHRrYZGBjwvv71r3vTp0/3srOzveuvv95rbW11t+hxcK7jcOTIEW/VqlVeQUGBFw6Hvfnz53vf/OY3ve7ubrcL/xDejgEA4MSEfw4IADA1UUAAACcoIACAExQQAMAJCggA4AQFBABwggICADhBAQEAnKCAAABOUEAAACcoIACAExQQAMCJ/w8FcaSVlZBayQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(images[0])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(28, 28, 1)"
      ]
     },
     "execution_count": 154,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "images[0].shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "targets = data[['akiec','bcc','bkl','df','mel','nv','vasc']].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    images, targets, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "8012"
      ]
     },
     "execution_count": 157,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2003"
      ]
     },
     "execution_count": 158,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "One of the dimensions in the output is <= 0 due to downsampling in conv2d_13. Consider increasing the input size. Received input shape [None, 1, 1, 32] which would produce output shape with a zero or negative value in a dimension.",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[164], line 3\u001b[0m\n\u001b[1;32m      1\u001b[0m EarlyStop \u001b[38;5;241m=\u001b[39m keras\u001b[38;5;241m.\u001b[39mcallbacks\u001b[38;5;241m.\u001b[39mEarlyStopping(min_delta\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0.001\u001b[39m, patience\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m10\u001b[39m, verbose\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0\u001b[39m)\n\u001b[0;32m----> 3\u001b[0m model \u001b[38;5;241m=\u001b[39m \u001b[43mkeras\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mSequential\u001b[49m\u001b[43m(\u001b[49m\u001b[43m[\u001b[49m\n\u001b[1;32m      4\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlayers\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mConv2D\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m16\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m3\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m3\u001b[39;49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mactivation\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mrelu\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43minput_shape\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43m(\u001b[49m\n\u001b[1;32m      5\u001b[0m \u001b[43m        \u001b[49m\u001b[38;5;241;43m28\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m28\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m1\u001b[39;49m\u001b[43m)\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m      6\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlayers\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mMaxPooling2D\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m      7\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlayers\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mBatchNormalization\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m      8\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlayers\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mConv2D\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m32\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m3\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m3\u001b[39;49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mactivation\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mrelu\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m      9\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlayers\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mMaxPooling2D\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m     10\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlayers\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mBatchNormalization\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m     11\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlayers\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mConv2D\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m32\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m3\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m3\u001b[39;49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mactivation\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mrelu\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m     12\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlayers\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mMaxPooling2D\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m     13\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlayers\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mBatchNormalization\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m     14\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlayers\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mConv2D\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m64\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m3\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m3\u001b[39;49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mactivation\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mrelu\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m     15\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlayers\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mMaxPooling2D\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m     16\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlayers\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mBatchNormalization\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m     17\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlayers\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mFlatten\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m     18\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlayers\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mDense\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m40\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mactivation\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mrelu\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m     19\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlayers\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mDropout\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m0.6\u001b[39;49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m     20\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlayers\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mDense\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m7\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mactivation\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43msoftmax\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m)\u001b[49m\n\u001b[1;32m     21\u001b[0m \u001b[43m]\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     23\u001b[0m model\u001b[38;5;241m.\u001b[39mcompile(\n\u001b[1;32m     24\u001b[0m     optimizer\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124madam\u001b[39m\u001b[38;5;124m'\u001b[39m,\n\u001b[1;32m     25\u001b[0m     loss\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mcategorical_crossentropy\u001b[39m\u001b[38;5;124m'\u001b[39m,\n\u001b[1;32m     26\u001b[0m     metrics\u001b[38;5;241m=\u001b[39m[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124maccuracy\u001b[39m\u001b[38;5;124m'\u001b[39m]\n\u001b[1;32m     27\u001b[0m )\n",
      "File \u001b[0;32m/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/tensorflow/python/trackable/base.py:204\u001b[0m, in \u001b[0;36mno_automatic_dependency_tracking.<locals>._method_wrapper\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m    202\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_self_setattr_tracking \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mFalse\u001b[39;00m  \u001b[38;5;66;03m# pylint: disable=protected-access\u001b[39;00m\n\u001b[1;32m    203\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m--> 204\u001b[0m   result \u001b[38;5;241m=\u001b[39m \u001b[43mmethod\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    205\u001b[0m \u001b[38;5;28;01mfinally\u001b[39;00m:\n\u001b[1;32m    206\u001b[0m   \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_self_setattr_tracking \u001b[38;5;241m=\u001b[39m previous_value  \u001b[38;5;66;03m# pylint: disable=protected-access\u001b[39;00m\n",
      "File \u001b[0;32m/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/keras/src/utils/traceback_utils.py:70\u001b[0m, in \u001b[0;36mfilter_traceback.<locals>.error_handler\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m     67\u001b[0m     filtered_tb \u001b[38;5;241m=\u001b[39m _process_traceback_frames(e\u001b[38;5;241m.\u001b[39m__traceback__)\n\u001b[1;32m     68\u001b[0m     \u001b[38;5;66;03m# To get the full stack trace, call:\u001b[39;00m\n\u001b[1;32m     69\u001b[0m     \u001b[38;5;66;03m# `tf.debugging.disable_traceback_filtering()`\u001b[39;00m\n\u001b[0;32m---> 70\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m e\u001b[38;5;241m.\u001b[39mwith_traceback(filtered_tb) \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m     71\u001b[0m \u001b[38;5;28;01mfinally\u001b[39;00m:\n\u001b[1;32m     72\u001b[0m     \u001b[38;5;28;01mdel\u001b[39;00m filtered_tb\n",
      "File \u001b[0;32m/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/keras/src/layers/convolutional/base_conv.py:354\u001b[0m, in \u001b[0;36mConv.compute_output_shape\u001b[0;34m(self, input_shape)\u001b[0m\n\u001b[1;32m    347\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m tf\u001b[38;5;241m.\u001b[39mTensorShape(\n\u001b[1;32m    348\u001b[0m             input_shape[:batch_rank]\n\u001b[1;32m    349\u001b[0m             \u001b[38;5;241m+\u001b[39m [\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mfilters]\n\u001b[1;32m    350\u001b[0m             \u001b[38;5;241m+\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_spatial_output_shape(input_shape[batch_rank \u001b[38;5;241m+\u001b[39m \u001b[38;5;241m1\u001b[39m :])\n\u001b[1;32m    351\u001b[0m         )\n\u001b[1;32m    353\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m:\n\u001b[0;32m--> 354\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[1;32m    355\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mOne of the dimensions in the output is <= 0 \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    356\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdue to downsampling in \u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mname\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m. Consider \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    357\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mincreasing the input size. \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    358\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mReceived input shape \u001b[39m\u001b[38;5;132;01m{\u001b[39;00minput_shape\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m which would produce \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    359\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124moutput shape with a zero or negative value in a \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    360\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdimension.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    361\u001b[0m     )\n",
      "\u001b[0;31mValueError\u001b[0m: One of the dimensions in the output is <= 0 due to downsampling in conv2d_13. Consider increasing the input size. Received input shape [None, 1, 1, 32] which would produce output shape with a zero or negative value in a dimension."
     ]
    }
   ],
   "source": [
    "EarlyStop = keras.callbacks.EarlyStopping(min_delta=0.001, patience=10, verbose=0)\n",
    "\n",
    "model = keras.Sequential([\n",
    "    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),\n",
    "    layers.MaxPooling2D(),\n",
    "    layers.BatchNormalization(),\n",
    "    layers.Conv2D(32, (3, 3), activation='relu'),\n",
    "    layers.MaxPooling2D(),\n",
    "    layers.BatchNormalization(),\n",
    "    layers.Conv2D(32, (3, 3), activation='relu'),\n",
    "    layers.MaxPooling2D(),\n",
    "    layers.BatchNormalization(),\n",
    "    layers.Conv2D(64, (3, 3), activation='relu'),\n",
    "    layers.MaxPooling2D(),\n",
    "    layers.BatchNormalization(),\n",
    "    layers.Flatten(),\n",
    "    layers.Dense(40, activation='relu'),\n",
    "    layers.Dropout(0.6),\n",
    "    layers.Dense(7, activation='softmax')\n",
    "])\n",
    "\n",
    "model.compile(\n",
    "    optimizer='adam',\n",
    "    loss='categorical_crossentropy',\n",
    "    metrics=['accuracy']\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train = model.fit(X_train, y_train,\n",
    "                  batch_size=32,\n",
    "                  epochs=100,\n",
    "                  callbacks=[EarlyStop],\n",
    "                  validation_data=(X_test, y_test))\n",
    "\n",
    "\n",
    "history_df = pd.DataFrame(train.history)\n",
    "history_df.loc[:, ['loss', 'val_loss']].plot()\n",
    "\n",
    "print('---------------------------------------------------------------------------------------')\n",
    "\n",
    "history_df = pd.DataFrame(train.history)\n",
    "history_df.loc[:, ['accuracy', 'val_accuracy']].plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "image = X_test[10]\n",
    "plt.imshow(image)\n",
    "plt.axis('off')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_image = image.reshape(1, 64, 64, 1)\n",
    "predictions = model.predict(test_image)\n",
    "print(predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# model.save(\"./weights/train_1.h5\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
