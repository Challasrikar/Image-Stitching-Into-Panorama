import cv2
import numpy as np
import os
import glob
import argparse
import copy


#Function to take commandline arguments as input
def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 2.")
    parser.add_argument(
        "--img_path", type=str, default="../extra1",
        help="path to the images")
    args = parser.parse_args()
    return args

#Function to read input images
def read_image(img_path, show=False):
    #This function is used to read the images from the directory
    img = cv2.imread(img_path)
    if not img.dtype == np.uint8:
        pass

    if show:
        show_image(img)

    img = [list(row) for row in img]
    return img

#Function to show images
def show_image(img, delay=1000):
    #This function is used to display the image
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()
    
#Function to write images 
def write_image(img, img_saving_path):
    #This function is used to write the output image to the file directory
    if isinstance(img, list):
        img = np.asarray(img, dtype=np.uint8)
    elif isinstance(img, np.ndarray):
        if not img.dtype == np.uint8:
            assert np.max(img) <= 1, "Maximum pixel value {:.3f} is greater than 1".format(np.max(img))
            img = (255 * img).astype(np.uint8)
    else:
        raise TypeError("img is neither a list nor a ndarray.")

    cv2.imwrite(img_saving_path, img)

#Function to detect the keypoints and descriptors of images using SIFT
def keypointsdetect(images_np):
    
    #Converting the images to greyscale
    gray_images=[cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images_np]
    
    #Converting the images to numpy arrays
    
    gray_images_np=[np.asarray(gray_image, dtype=np.uint8) for gray_image in gray_images]
    
    #print("length of gray_images",len(gray_images))
    #print("len of list image_np",len(images_np))
    #print("len of list gray_images_np",len(gray_images_np))
    #print("length of images_np",images_np[1].shape)
    #print("length of gray_np",gray_images_np[0].shape)
    
    sift=cv2.xfeatures2d.SIFT_create()
    
    keypoints=[]
    descriptors=[]
    
    for gray_image in gray_images_np:
        #The keypoints and descriptors for each image are found by using the function detectAndCompute
        keypoint, descriptor=sift.detectAndCompute(gray_image,None)
        
        #print("descriptor for each image",len(descriptor));
        
        #Converting the descriptors to numpy array
        np_desc=np.array(descriptor)
        
        #print("np_desc shape is ",np_desc.shape)
        
        #Appending all the obtained kepoints and descriptors for each image to their corresponding lists
        keypoints.append(keypoint)
        descriptors.append(np_desc)
    
    #print(len(keypoints))
    #print(len(descriptors))
    #print(type(descriptors))
    
    #Drawing an image that depicts the keypoints in both the images and matches them
    #sift_image=[cv2.drawKeypoints(gray_images_np[i],keypoints[i],images_np[i],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) for i in range(len(keypoints))]
    sift_image=[cv2.drawKeypoints(gray_images_np[i],keypoints[i],None,color=(0,255,0)) for i in range(len(keypoints))]
    
    #print("len of sift_images is",(len(sift_image)))
    
    #Returning the images with matched keypoints(not required for panorama, but added for clarity), keypoints and descriptors
    return sift_image, keypoints, descriptors

#Function to match the features that are obtained using SIFT
#The distances between all the features are calculated and the 2 matches that have the least distances are considered. 
#Ref: http://cs.brown.edu/courses/cs143/2013/results/proj2/rroelke/
def featurematching2(features_1,features_2):
    
    #print(type(features_1))
    #print(features_1.shape)
    
    n_f1=features_1.shape[0]
    n_f2=features_2.shape[0]
    matches=[]
    queries=[]
    trains=[]
    match_distances=[]
    distances=np.zeros((n_f1,n_f2))
    
    #Distance between all the features are calculated
    for i in range(n_f1):
        for j in range(n_f2):
            distances[i][j]=np.linalg.norm(features_1[i]-features_2[j])
    
    #These distances are sorted
    sorted_distances = np.argsort(distances,axis=1)
    
    #print("type of sorted distance is ",type(sorted_distances))
    #print("shape of sorted distance is ",sorted_distances.shape)
    
    #k=2 implies that the 2 nearest neighbours are considered.
    k=2
    for i in range(n_f1):
        neighbours=sorted_distances[i][:k]
        
        #print("type of neighbours is ",type(neighbours))
        #print("length of neighbours is ",len(neighbours))
        #print(neighbours)
        
        temp_matches=[]
        temp_queries=[]
        temp_trains=[]
        temp_distances=[]
        
        #Here, k value is fixed as 2. But, for convenience, for loop is used.
        for j in range(k):
            
            #For each neighbour the query_index, train_index and distances are calculated.
            query_index=i;
            train_index=neighbours[j]
            distance=distances[i][neighbours[j]]
            
            temp_queries.append(query_index)
            temp_trains.append(train_index)
            temp_distances.append(distance)
            
            #print("length of temp is ",len(temp))
            #print("temp in for loop is",temp)
        
        #print("temp outside for loop is",temp)
        
        #All the values that are obtained are appended to their corresponding lists. 
        queries.append(temp_queries)
        trains.append(temp_trains)
        match_distances.append(temp_distances)
        
        #print("matches inside for loop is", matches)
        
    #print("matches outside for loop is", matches)
    
    #print(type(matches))
    #print(len(matches))
    
    #print(matches[0][0].distance)
    #print(matches[0][1].distance)
    #print(match_distances[0][0])
    #print(match_distances[0][1])
    
    good_queries=[]
    good_trains=[]
    good_index=[]
    good_distances=[]
    threshold=0.75
    
    #for i,j in matches:
        #print("i distance is ",i.distance)
        #print("j distance is ",j.distance)
        
        #if (i.distance/j.distance)<threshold:
            #good_matches.append(i)
    
    for i in match_distances:
        if(i[0]<(threshold*i[1])):
            #print(match_distances.index(i))
            
            good_index.append(match_distances.index(i))
    
    for i in good_index:
        good_queries.append(queries[i][0])
        good_trains.append(trains[i][0])
        good_distances.append(match_distances[i][0])
        
        #print("match distance i is",i)
        #print("match distance j is",j)
        
    #print(len(good_distances))
    #print("type of good_matches",type(good_matches))
    #print("length of good matches",len(good_matches))
    #print(good_matches)
    
    #print("len of good_queries",len(good_queries))
    #print("printing good queries",good_queries)
    #print(good_queries[0],good_trains[0],good_distances[0])
    
    #The  three different lists are zipped together and converted to a set
    good_matches_set=set(zip(good_queries,good_trains,good_distances))
    
    #print(len(good_matches_set))
    
    #The obtained set is converted to list
    good_matches=[*good_matches_set,]
    
    #print(len(good_matches))
    #print(good_matches[0])
    
    #The obtained list is sorted
    good_matches.sort(key = lambda x: x[0])
    
    #print("after sorting matches")
    #print(len(good_matches))
    #print(good_matches[0])
    
    #Finally, the good matches that are obtained will be returned.
    return good_matches

#Function to create a homography matrix
# Ref: http://www.csc.kth.se/~perrose/files/pose-init-model/node17.html
def get_homography_matrix(query_points, train_points):
    point_pairs=[]
    for i in range(len(query_points)):
        point_pairs.append([query_points[i],train_points[i]])
    
    #4 points in each image are obtained and these are represented as a 8 x 9 matrix, P
    P_list=[]
    for a,b in point_pairs:
        x1,y1=a[0],a[1]
        x2,y2=b[0],b[1]
        
        P_list += [[-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2],
                       [0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2]]
    
    #print(P_list)
    
    #The list of list is converted to numpy array
    P=np.array(P_list)
    #print(P.shape)
    
    #The required homography matrix is the eigen vector corresponding to the least eigen value for the Matrix |P.T*P|
    #First the matrix product P.T.P is calculated, where P.T is the transpose matrix of P
    M = np.dot(P.T, P)
    
    #The eigen values and the eigen vectors are calculated for that matrix.
    eigen_values, eigen_vectors = np.linalg.eig(M)
    
    #The minimum eigen value is determined
    min_eigen=np.argmin(eigen_values)
    
    #The eigen vector corresponding to the minimum eigen value is calculated.
    eig_vec=eigen_vectors[:,min_eigen]
    
    #This eigen vector is normalized
    normal=np.linalg.norm(eig_vec)
    eig_vec=eig_vec/normal
    
    #The required homography matrix is of the shape (3,3). Hence, the eigen vector is reshaped.
    H = eig_vec.reshape(3, 3)
    
    return H;


#Function that returns the best matches and best homography matrix by iterating on different samples of matches.
def RANSAC(matches, keypoints_1, keypoints_2, batch_size=4, iterations =50):
    
    #print("In RANSAC: printing type(matches)",type(matches))
    #print("In RANSAC: printing type(matches[0])",type(matches[0]))
    #print(matches[0])
    
    #The threshold is considered to be 5.0
    reproj_threshold = 5.0
    
    best_count = -1
    best_H_matrix = []
    best_matches = []
    good_matches = []
    for itr in range(iterations):
        np.random.shuffle(matches)
        good_matches = []
        #print("match 1 after shuffling",matches[0])
        
        #A batch of matches are considered for each iteration. 
        for i in range(0,len(matches), batch_size):
            qp=[]
            tp=[]
            
            #For each match, the coordinates of the keypoints are extracted
            for m in matches[i:i+batch_size]:
                qp.append(keypoints_1[m[0]].pt)
                tp.append(keypoints_2[m[1]].pt)
            
            #These coordinates are fed to the get_homography_matrix to obtain the homography matrix.
            H = get_homography_matrix(qp,tp)
            
            count=0
            good_matches = []
            
            #For each match, the error is calculated between the actual point and the point predicted using the homography matrix.
            for m in matches:    
                x1 = keypoints_1[m[0]].pt[0]
                y1 = keypoints_1[m[0]].pt[1]
                
                x2 = keypoints_2[m[1]].pt[0]
                y2 = keypoints_2[m[1]].pt[1]
                
                pred = np.dot(H, [x1, y1, 1])
                pred = pred/pred[-1]
                predicted_point = pred[:-1]
                
                actual_point = [x2, y2]
                
                #Calculating the error between the predcicted point and the actual point 
                error = np.linalg.norm(np.subtract(actual_point,predicted_point))
                
                #If the error obtained is less than the threshold, then the corresponding match is added to the good_matches list
                if error<reproj_threshold:
                    good_matches.append(m)
                    count+=1
            
            #The number of inliers are calculated. If they are greater than the previous iteration, then the values are updated.
            if count>best_count:
                best_count = count
                best_H_matrix=H
                best_matches=good_matches
                
                #If the number of inliers are greater than one-third of the total keypoints, then the value is returned.
                if best_count>len(keypoints_1)/3:
                    best_H_matrix = best_H_matrix/best_H_matrix[-1,-1]
                    return best_H_matrix, best_matches
    
    #Else, the best values obtained are returned.    
    best_H_matrix = best_H_matrix/best_H_matrix[-1,-1]        
    return best_H_matrix, best_matches;

#Function to warp two images.
def warping_images(image_1, image_2, H):
    
    #The two obtained images are converted to numpy arrays
    image_1 = np.asarray(image_1, dtype=np.uint8)
    image_2 = np.asarray(image_2, dtype=np.uint8)
    
    #print("shape of image 1 is",image_1.shape)
    #print("shape of image 2 is",image_2.shape)
    
    height_1 = image_1.shape[0]
    width_1 = image_1.shape[1]
    height_2 = image_2.shape[0]
    width_2 = image_2.shape[1]
    
    #The points from each image is added to their corresponding numpy array
    points_in_image_1 = np.float32([[0, 0], [0, height_1], [width_1, height_1],[width_1, 0]]).reshape(-1,1,2)
    points_in_image_2 = np.float32([[0, 0], [0, height_2], [width_2, height_2],[width_2, 0]]).reshape(-1,1,2)
    
    #Calculates the perspective transform
    transform_points = cv2.perspectiveTransform(points_in_image_2, H)
    
    points = np.concatenate((points_in_image_1, transform_points), axis=0)
    x_min, y_min = np.int32(points.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(points.max(axis=0).ravel() + 0.5)
    
    #The translation is determined
    translation_point = [-x_min, -y_min]
    
    M = np.array([[1, 0, translation_point[0]],
                  [0, 1, translation_point[1]],
                  [0, 0, 1]])
    
    #The two images are stitched together
    panorama = cv2.warpPerspective(image_1, np.dot(M, H), (x_max - x_min, y_max - y_min))
    #print((panorama.shape))
    panorama[translation_point[1]:height_2 + translation_point[1], translation_point[0]: width_2 + translation_point[0]] = image_2
    
    return panorama

#Function to concatenate two images, when there are no matches between them.
def concat_images(image_1, image_2):
    
    y1,x1 = image_1.shape[:2]
    y2,x2 = image_2.shape[:2]
    
    max_height = np.max([y1, y2])
    total_width = x1 +x2
    
    new_image = np.zeros((max_height, total_width, 3))
    new_image[:y1,:x1]=image_1
    new_image[:y2,x1:x1+x2]=image_2
    
    print(type(new_image))
    
    return new_image

#Function to stitch two images.
def stitch(images):
    
    #Step 1: SIFT for all given images
    sift_images,keypoints, descriptors=keypointsdetect(images)
    
    #print(type(descriptors))
    #print(type(descriptors[0]))
    #print(descriptors[0].shape)
    
    #Step 2: Feature Matching
    matches=featurematching2(descriptors[0],descriptors[1])
    
    #Step 3: Getting Homography matrix and matches using RANSAC
    H_matrix, matches=RANSAC(matches, keypoints[0],keypoints[1])
    
    #Printing the homography matrix
    #print(H_matrix)
    if len(matches)<10:
        #The images donot match, they need to be displayed side-by-side
        #print("Photos donot overlap")
        
        result=np.asarray(concat_images(images[0], images[1]), dtype=np.uint8)
        
        #print(result.shape)
        
    else:
        #Step 4: Image Warping and stitching
        result = warping_images(images[0], images[1], H_matrix)
            
        #Step 5: Removing the black region from the image
        result=remove_black_region(result)
    
    #Printing the input images
    #count=0
    #for img in images:
        #write_image(img,"./results/input_"+str(count)+".jpg")
        #count+=1
        
    #Printing the sift images    
    #sift_count=0
    #for img in sift_images:
        #write_image(img,"./results/sift_image_"+str(sift_count)+".jpg")
        #sift_count+=1
        
    #print("write success")
    
    return result

#Function to remove the black pixels in the resultant image being formed
def remove_black_region(image):
    # Mask of non-black pixels (assuming image has a single channel).
    mask = image > 0

    # Coordinates of non-black pixels.
    points = np.argwhere(mask)

    # Bounding box of non-black pixels.
    x1, y1 = points.min(axis=0)[:2]
    x2, y2 = points.max(axis=0)[:2] + 1   

    # Get the contents of the bounding box.
    cropped_image = image[x1:x2, y1:y2]
    return cropped_image
     
#Function to calclate the distances between all the matches between the images
def calculate_distances(image_1, image_2):
    
    #Getting the best matches for both the images
    sift_images,keypoints, descriptors=keypointsdetect([image_1, image_2])
    matches=featurematching2(descriptors[0],descriptors[1])
    H_matrix, matches=RANSAC(matches, keypoints[0],keypoints[1])
    
    #Calculating the distance between those matches and storing them in a list.
    distances=[]
    for match in matches:
        point_1 = np.array(keypoints[0][match[0]].pt)
        point_2 = np.array(keypoints[1][match[1]].pt)
        
        point_2[0] += image_2.shape[1]
        
        distance = np.linalg.norm(point_1 - point_2)
        distances.append(distance)
    return distances;

#Function to find the order of the two images
def get_ordering(image_1, image_2):
    
    #Calculating the distances of matches in the given order
    test_given = calculate_distances(image_1, image_2)
    
    #Calculating the distances of matches in the reverse order
    test_reverse = calculate_distances(image_2, image_1)
    
    #If the average of the distances for the given order is less than reverse, then the given order is the correct order for those images.
    if(np.mean(test_reverse)>np.mean(test_given)):
        return -1
    
    #Else, the the images are in the reverse order and need to be swapped 
    return 1

#Main function
def main():
    
    args = parse_args()
    
    dir = args.img_path
    #Reading the images from directory
    filenames = glob.glob(os.path.join(dir,"*.jpg"))
    filenames.sort()
    images=[]
    images = [read_image(img) for img in filenames]
    
    print("Image read")
    print(len(images))
    if(len(images)==0):
        print("no images found")
        exit(0)
        
    images=[np.asarray(image, dtype=np.uint8) for image in images]
    #images[0]=np.asarray(images[0], dtype = np.uint8)
    #images[1]=np.asarray(images[1], dtype = np.uint8)
    
    #print((images[0].shape))
    #print((images[1].shape))
    
    if(len(images)==1):
        result=images[0]
    elif(len(images)==2):
        result=stitch([images[0], images[1]])
        write_image(result,os.path.join(dir,"panorama.jpg"))
    else:
        #Step 0: Ordering the given images 
        for i in range(len(images)):
            for j in range(len(images)):
                if(i!=j):
                    if(get_ordering(images[i],images[j])==1):
                        temp=images[i]
                        images[i]=images[j]
                        images[j]=temp
    
        print("ordering done")
        result=stitch([images[0], images[1]])
        for i in range(2,len(images)):
            result=stitch([result, images[i]])
        write_image(result,os.path.join(dir,"panorama.jpg"))
    
    print("panorama done")
    
    
if __name__ == "__main__":
    main()
