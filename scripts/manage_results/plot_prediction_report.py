from meld_classifier.experiment import Experiment
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
from meld_classifier.dataset import load_combined_hemisphere_data, Dataset, normalise_data
from meld_classifier.meld_plotting import trim,rotate90
from meld_classifier.evaluation import Evaluator
from meld_classifier.paths import (
    BASE_PATH,
    MELD_DATA_PATH,
    DK_ATLAS_FILE,
    EXPERIMENT_PATH, 
    MODEL_PATH,
    MODEL_NAME,
    FINAL_SCALING_PARAMS,
    SURFACE_PARTIAL, 
    DEFAULT_HDF5_FILE_ROOT,
    SCRIPTS_DIR,
)
import os
import json
import glob
import h5py
import argparse
import numpy as np
import nibabel as nb
from nilearn import plotting, image
import seaborn as sns
import pandas as pd
import matplotlib_surface_plotting as msp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image
import subprocess
import meld_classifier.mesh_tools as mt
import meld_classifier.paths as paths
from datetime import date
from fpdf import FPDF

class PDF(FPDF):    
    def lines(self):
        self.set_line_width(0.0)
        self.line(5.0,5.0,205.0,5.0) # top one
        self.line(5.0,292.0,205.0,292.0) # bottom one
        self.line(5.0,5.0,5.0,292.0) # left one
        self.line(205.0,5.0,205.0,292.0) # right one
    
    def custom_header(self, logo, txt1, txt2=None):
        # Log
        self.image(logo, 10, 8, 33)
        # Arial bold 
        self.set_font('Arial', 'B', 30)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(w=30, h=10, txt=txt1, border=0, ln=0, align='C')
        if txt2 != None:
            # Arial bold 15
            self.ln(20)
            self.cell(80)
            self.set_font('Arial', 'B', 20)
            self.cell(w=30, h=5, txt=txt2, border=0, ln=0, align='C')
        # Line break
        self.ln(20)
    
    def custom_footer(self, txt):
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Position at 1.5 cm from bottom
        self.set_y(-30)
        # add text
        self.cell(w=0, h=0, txt=txt, border=0, ln=2, align='C') 
        # Date
        today = date.today()
        today = today.strftime("%d/%m/%Y")
        self.cell(w=5, h=8, txt=str(today) , border=0, ln=0, align='L')
        # Page number
        self.cell(w=180, h=8, txt='Page ' + str(self.page_no()), border=0, ln=2, align='R')
                
    
    def info_box(self, txt):
        # set font
        self.set_font('Arial', 'I', 10)
        #set box color
        self.set_fill_color(160,214,190)
        # add texte box info
        self.multi_cell(w=190, h=5, txt=txt , border=1, align='L', fill=True)
        
    def info_box_clust(self, txt):
        self.ln(140)
        # set font
        self.set_font('Arial', 'I', 6)
        #set box color
        self.set_fill_color(160,214,190)
        # add text box info
        self.multi_cell(w=80, h=5, txt=txt , border=1, align='L', fill=True)
    
    def disclaimer_box(self, txt):
        # set font
        self.set_font('Arial', 'I', 6)
        #set box color
        self.set_fill_color(240,128,128)
        # add texte box info
        self.multi_cell(w=190, h=5, txt=txt , border=1, align='L', fill=True)
        
    def subtitle_inflat(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 20)
        # Title
        self.cell(w=10, h=80, txt='Overview clusters on inflated brain', border=0, ln=2, align='L')
       
    def imagey(self,im, y):
        pdf.image(im, 5, y, link='', type='', w=190, h=297/3)

        
def load_prediction(subject,hdf5):
    results={}
    with h5py.File(hdf5, "r") as f:
        for hemi in ['lh','rh']:
            results[hemi] = f[subject][hemi]['prediction'][:]
    return results

def create_surface_plots(surf,prediction,c):
    """plot and reload surface images"""
    
    msp.plot_surf(surf['coords'],
                                           surf['faces'],prediction,
              rotate=[90],
              mask=prediction==0,pvals=np.ones_like(c.cortex_mask),
              colorbar=False,vmin=0,vmax=1,cmap='rainbow',
              base_size=20,
              filename='tmp.png'
             );
#    subprocess.call(f"convert ./tmp.png -trim ./tmp1.png", shell=True)
    im = Image.open('tmp.png')
    im = trim(im)
    im = im.convert("RGBA")
    im1 = np.array(im)
    msp.plot_surf(surf['coords'],
                                           surf['faces'],prediction,
              rotate=[270],
              mask=prediction==0,pvals=np.ones_like(c.cortex_mask),
              colorbar=False,vmin=0,vmax=1,cmap='rainbow',
              base_size=20,
              filename='tmp.png'
             );
 #   subprocess.call(f"convert ./tmp.png -trim ./tmp1.png", shell=True)
    im = Image.open('tmp.png')
    im = trim(im)
    im = im.convert("RGBA")
    im2 = np.array(im)
    plt.close('all')
    return im1,im2

def load_cluster(file, subject):
    df=pd.read_csv(file,index_col=False)
    n_clusters = df[df['ID']==subject]['n_clusters']
    return np.array(n_clusters)[0]

def get_key(dic, val):
    # function to return key for any value in dictionnary
    for key, value in dic.items():
        if val == value:
            return key
    return "No key for value {}".format(val)

def define_atlas():
    atlas = nb.freesurfer.io.read_annot(os.path.join(BASE_PATH, DK_ATLAS_FILE))
    vertex_i = np.array(atlas[0]) - 1000  # subtract 1000 to line up vertex
    rois_prop = [
        np.count_nonzero(vertex_i == x) for x in set(vertex_i)
    ]  # proportion of vertex per rois
    rois = [x.decode("utf8") for x in atlas[2]]  # extract rois label from the atlas
    rois = dict(zip(rois, range(len(rois))))  # extract rois label from the atlas
    rois.pop("unknown")  # roi not part of the cortex
    rois.pop("corpuscallosum")  # roi not part of the cortex
    return rois, vertex_i, rois_prop

def get_cluster_location(cluster_array):
    cluster_array = np.array(cluster_array)
    rois, vertex_i, rois_prop = define_atlas()
    pred_rois = list(vertex_i[cluster_array])
    pred_rois = np.array([[x, pred_rois.count(x)] for x in set(pred_rois) if x != 0])
    ind = pred_rois[np.where(pred_rois == pred_rois[:,1].max())[0]][0][0]
    location = get_key(rois,ind)
    return location

def save_mgh(filename, array, demo):
    """save mgh file using nibabel and imported demo mgh file"""
    mmap = np.memmap("/tmp/tmp", dtype="float32", mode="w+", shape=demo.get_data().shape)
    mmap[:, 0, 0] = array[:]
    output = nb.MGHImage(mmap, demo.affine, demo.header)
    nb.save(output, filename)
    
if __name__ == "__main__":
    # Set up experiment
    parser = argparse.ArgumentParser(description="create mgh file with predictions from hdf5 arrays")
    parser.add_argument(
        "--experiment-folder",
        help="Experiments folder",
    )
    parser.add_argument(
        "--experiment-name",
        help="subfolder to use, typically the ensemble model",
        default="ensemble_iteration",
    )
    parser.add_argument("--fold", default=None, help="fold number to use (by default all)")
    parser.add_argument(
        "--subjects_dir", default="", help="folder containing freesurfer outputs. It will store predictions there"
    )
    parser.add_argument("--list_ids", default=None, help="texte file containing list of ids to process")
    args = parser.parse_args()
     
     
    # setup parameters
    base_feature_sets = ['.on_lh.gm_FLAIR_0.5.sm10.mgh',
                         '.on_lh.wm_FLAIR_1.sm10.mgh',
                         '.on_lh.curv.sm5.mgh',
                         '.on_lh.pial.K_filtered.sm20.mgh',
                         '.on_lh.sulc.sm5.mgh',
                         '.on_lh.thickness.sm10.mgh',
                         '.on_lh.w-g.pct.sm10.mgh',
                         ]

    feature_names_sets = ['GM FLAIR (50%)',
                         'WM FLAIR (1mm)',
                         'Mean curvature',
                         'Intrinsic Curvature',
                         'Sulcal depth',
                         'Cortical thickness',
                         'Grey-white contrast',]
    
    subjects_dir = args.subjects_dir
    subjids = np.loadtxt(args.list_ids, dtype="str", ndmin=1)
    exp_path = os.path.join(MELD_DATA_PATH, args.experiment_folder)
    hemis = ["lh", "rh"]
    c = MeldCohort(hdf5_file_root=DEFAULT_HDF5_FILE_ROOT)
    surf = mt.load_mesh_geometry(os.path.join(paths.BASE_PATH,SURFACE_PARTIAL))
    
    # select predictions files
    if args.fold == None: 
        hdf_predictions = os.path.join(exp_path, "results", f"predictions_{args.experiment_name}.hdf5")
        csv_results = os.path.join(exp_path, "results", f"test_results_.csv")
    else : 
        hdf_predictions = os.path.join(exp_path, f"fold_{args.fold}", "results", f"predictions_{args.experiment_name}.hdf5")
        csv_results = os.path.join(exp_path, f"fold_{args.fold}", "results", f"test_results_.csv")
   
    # Provide models parameter
    exp = Experiment(experiment_path=os.path.join(EXPERIMENT_PATH, MODEL_PATH), experiment_name=MODEL_NAME)
    saliency_features = exp.get_features()[0]
    eva = Evaluator(exp)

    for subject in subjids:
        print(subject)
        subj = MeldSubject(subject, cohort=c)
        subject_dir = os.path.join(subjects_dir,'input',subject)
        #create output directory
        output_dir = os.path.join(subject_dir,'predictions','reports')
        os.makedirs(output_dir , exist_ok=True)           
        # load predictions subject
        result_hemis = load_prediction(subject,hdf_predictions)
        number_cluster =  load_cluster(csv_results, subject)      
        # Open their MRI data if available
        t1_file = glob.glob(os.path.join(subject_dir,'T1','*T1*.nii*'))[0]
        prediction_file_lh = glob.glob(os.path.join(subject_dir,'predictions','lh.prediction*'))[0]
        prediction_file_rh = glob.glob(os.path.join(subject_dir,'predictions','rh.prediction*'))[0]
        # load image
        imgs={'anat' : nb.load(t1_file),
              'pred_lh' : nb.load(prediction_file_lh), 
              'pred_rh' : nb.load(prediction_file_rh)}    
        # Resample and move to same shape and affine than t1
        imgs['pred_lh'] = image.resample_img(imgs['pred_lh'], target_affine=imgs['anat'].affine, target_shape=imgs['anat'].shape, 
                                             interpolation='nearest', copy=True, 
                                             order='F', clip=False, fill_value=0, force_resample=False)
        imgs['pred_rh'] = image.resample_img(imgs['pred_rh'], target_affine=imgs['anat'].affine, target_shape=imgs['anat'].shape, 
                                             interpolation='nearest', copy=True, 
                                             order='F', clip=False, fill_value=0, force_resample=False)                
        #initialise parameter for plot
        fig= plt.figure(figsize=(15,8), constrained_layout=True)
        features=subj.get_feature_list()
        if 'FLAIR' in features:
            base_features=base_feature_sets
            feature_names=feature_names_sets
        else :
            base_features=base_feature_sets[2:]
            feature_names=feature_names_sets[2:]
        labels_hemis = {}
        features_hemis={}
        predictions={}
        clusters={}
        list_clust={}
        # Loop over hemi
        for i, hemi in enumerate(['lh','rh']):
            #prepare grid plot
            gs1 = GridSpec(2, 3, width_ratios=[1, 1, 1],  wspace=0.1, hspace=0.1)
            gs2 = GridSpec(1, 2, width_ratios=[1, 3], wspace=2)
            gs3 = GridSpec(1, 1)
            #get features array
            features_hemis[hemi], labels_hemis[hemi] = subj.load_feature_lesion_data(features, hemi=hemi, features_to_ignore=[])
            features_hemis[hemi]=normalise_data(features_hemis[hemi],features,FINAL_SCALING_PARAMS)
            #get prediction
            predictions[hemi] = np.zeros(len(c.cortex_mask))
            predictions[hemi][c.cortex_mask] = result_hemis[hemi]
            #plot predictions on inflated brain
            im1,im2=create_surface_plots(surf,prediction=predictions[hemi],c=c)
            if hemi=='rh':
                im1 = im1[:,::-1]
                im2 = im2[:,::-1]
            ax = fig.add_subplot(gs1[i,1])
            ax.imshow(im1)
            ax.axis('off')
            ax.set_title(hemi, loc='left', fontsize=20)
            ax = fig.add_subplot(gs1[i,2])
            ax.imshow(im2)
            ax.axis('off')
            # load gradients for saliencies 
            with h5py.File(hdf_predictions, 'r') as f:
                igp=f[subject][hemi]['integrated_gradients_pred'][:]
            #initiate params for saliencies
            prefixes = ['.combat','.inter_z.intra_z.combat',
                       '.inter_z.asym.intra_z.combat']
            names=['combat','norm', 'asym']
            lims = np.max([-0.1,0.1])
            norm = mpl.colors.Normalize(vmin=-lims, vmax=lims)
            cmap = mpl.colors.LinearSegmentedColormap.from_list('grpr',colors=['#276419','#FFFFFF','#8E0152',])
            m = cm.ScalarMappable(norm=norm, cmap=cmap)
            labels=['Combat','Normalised','Asymmetry']
            hatching = ['\\\\','//','--']
            list_clust[hemi] = list(set(predictions[hemi]))
            list_clust[hemi].remove(0)
            #loop over clusters
            for cluster in list_clust[hemi]:
                print(cluster)
                fig2 = plt.figure(figsize=(17,9))
                ax2 = fig2.add_subplot(gs2[0, 1], )
                # get and plot saliencies
                for pr,prefix in enumerate(prefixes):
                    cur_data = np.zeros(len(base_features))
                    cur_err = np.zeros(len(base_features))
                    saliency_data = np.zeros(len(base_features))
                    for b,bf in enumerate(base_features):
                        cur_data[b] = np.mean(features_hemis[hemi][predictions[hemi]==cluster,features.index(prefix+bf)])
                        cur_err[b] = np.std(features_hemis[hemi][predictions[hemi]==cluster,features.index(prefix+bf)])
                        saliency_data[b] = np.mean(igp[predictions[hemi][c.cortex_mask]==cluster,saliency_features.index(prefix+bf)])
                    ax2.barh(y=np.array(range(len(base_features)))-pr*0.3, width=cur_data,hatch= hatching[pr],
                                 height=0.3, edgecolor='k',xerr=cur_err,label=labels[pr],
                           color=m.to_rgba(saliency_data))
                ax2.set_xlim([-8,8])
                ax2.set_xticks([])
                ax2.set_yticks(np.array(range(len(base_features)))-0.23)    ;                          
                ax2.set_yticklabels(feature_names, fontsize=16)    ; 
                ax2.set_xlabel('Z score', fontsize=16)
                ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), fontsize=16)
                fig2.colorbar(m,label=f'Saliency',ax=ax2,ticks=[-0.1,-0.05,0,0.05,0.1])
                ax2.set_autoscale_on(True)
                ## display info cluster
                # get size
                size_clust = np.sum(c.surf_area[predictions[hemi]==cluster])/100
                size_clust = round(size_clust, 3)
                # get location
                location = get_cluster_location(predictions[hemi]==cluster)
                #plot info in text box in upper left in axes coords
                textstr = '\n'.join((
                f' cluster {int(cluster)} on {hemi} hemi', ' ',
                f' size cluster = {size_clust} cm2', ' ',
                f' location =  {location}'))
                props = dict(boxstyle='round', facecolor='gray', alpha=0.5)
                ax2 = fig2.add_subplot(gs2[0, 0])
                ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=18,
                        verticalalignment='top', bbox=props)
                ax2.axis('off')
#                 fig2.tight_layout()
                fig2.savefig(f'{output_dir}/saliency_{subject}_{hemi}_c{int(cluster)}.png')
                #plot cluster on anat MRI volume
                fig3= plt.figure(figsize=(15,8))
                ax3 =fig3.add_subplot(gs3[0])
                min_v = cluster-1
                max_v = cluster+1
                mask = image.math_img(f"(img < {max_v}) & (img > {min_v})", img= imgs[f'pred_{hemi}'])
                coords = plotting.find_xyz_cut_coords(mask)
                vmax = np.percentile(imgs['anat'].get_fdata(), 99)
                display = plotting.plot_anat(t1_file, colorbar=False, cut_coords=coords, draw_cross= True,
                                             figure=fig3, axes = ax3,  vmax = vmax)
                display.add_contours(prediction_file_lh, filled=True, alpha=0.7, levels=[0.5], colors='red')
                display.add_contours(prediction_file_rh, filled=True, alpha=0.7, levels=[0.5], colors='red')
                # save figure for each cluster
#                 fig3.tight_layout()
                fig3.savefig(f'{output_dir}/mri_{subject}_{hemi}_c{int(cluster)}.png')
        # Add information subject in text box
        n_clusters = len(list_clust['lh'])  + len(list_clust['rh'])   
        ax = fig.add_subplot(gs1[0, 0])
        textstr = '\n'.join((
        f'patient {subject}',
        ' ',
        f'number of predicted clusters = {n_clusters}'))
        # place a text box in upper left in axes coords
        props = dict(boxstyle='round', facecolor='gray', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=16,
                verticalalignment='top', bbox=props)
        ax.axis('off')
        # save overview figure
        fig.savefig(f'{output_dir}/inflatbrain_{subject}.png')
        
        
        # create PDF overview
        pdf = PDF()#pdf object
        pdf=PDF(orientation='P') # landscape
        pdf=PDF(unit='mm') #unit of measurement
        pdf=PDF(format='A4') #page format. A4 is the default value of the format, you don't have to specify it.


        logo = os.path.join(SCRIPTS_DIR,'MELD_logo.png')

        text_info_1 = "Information: \n The MRI data of this patient has been processed through the MELD surface-based FCD detection algorithm. \n Page 1 of this report will show all detected clusters on an inflated view of the brain. \n Each subsequent page is an individual cluster"

        text_info_2 = "For each cluster, the information below are provided: \n   -The hemisphere the cluster is on \n   -The surface area of the cluster (across the cortical surface) \n   -The location of the cluster \n   -The z-scores of the patient cortical features averaged within the cluster. \n   -The saliency of each feature to the network - if a feature is brighter pink, that feature was more important to the network. \n \n For more information, please read the Guide to using the MELD surface-based FCD detection."

        disclaimer = "Disclaimer: The MELD surface-based FCD detection algorithm is intended for research purposes only and has not been reviewed or approved by the Medicines and Healthcare products Regulatory Agency (MHRA),European Medicine Agency (EMA) or by any other agency. Any clinical application of the software is at the sole risk of the party engaged in such application. \nThere is no warranty of any kind that the software will produce useful results in any way. Use of the software is at the recipient own risk" 

        footer_txt="This report was created by Mathilde Ripart, Hannah Spitzer, Sophie Adler and Konrad Wagstyl on behalf of the MELD Project"

        today = date.today()

        #### create main page with overview on inflated brain

        #add page
        pdf.add_page()
        #add line contours
        pdf.lines()
        #add header
        pdf.custom_header(logo, txt1='MELD report', txt2=f'Patient ID: {subject}')
        #add info box
        pdf.info_box(text_info_1)
        #add disclaimer box
        pdf.disclaimer_box(disclaimer)
        #add image
        pdf.subtitle_inflat()
        im_inflat= os.path.join(output_dir, f'inflatbrain_{subject}.png')
        pdf.imagey(im_inflat,130)
        #add footer date
        pdf.custom_footer(footer_txt)

        clusters=range(1,n_clusters+1)
        #### Create page for each cluster with MRI view and saliencies
        for cluster in clusters: 
            #add page
            pdf.add_page()
            #add line contours
            pdf.lines()
            #add header
            pdf.custom_header(logo, txt1='MRI view & saliencies', txt2=f'Cluster {cluster}')
            #add image
            im_mri= glob.glob(os.path.join(output_dir, f'mri_{subject}_*_c{cluster}.png'))[0]
            pdf.imagey(im_mri, 50)
            #add image
            im_sal= glob.glob(os.path.join(output_dir, f'saliency_{subject}_*_c{cluster}.png'))[0]
            pdf.imagey(im_sal, 150)
            #add info cluster analysis txt only 1st cluster
            if cluster==1:
                pdf.info_box_clust(text_info_2)
            #add footer date
            pdf.custom_footer(footer_txt)
        #save pdf
        pdf.output(os.path.join(output_dir,f'MELD_report_{subject}.pdf'),'F')