import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from panoptes_client import Subject
from skimage import io, transform

##Modified from the ramanakumars/SolarJets/ directory 

def get_box_edges(x, y, w, h, a):
    '''
        Return the corners of the box given one corner, width, height
        and angle

        Inputs
        ------
        x : float
            Box left bottom edge x-coordinate
        y : float
            Box left bottom edge y-coordinate
        w : float
            Box width
        h : float
            Box height
        a : flat
            Rotation angle

        Outputs
        --------
        corners : numpy.ndarray
            Length 4 array with coordinates of the box edges
    '''
    cx = (2*x+w)/2
    cy = (2*y+h)/2
    centre = np.array([cx, cy])
    original_points = np.array(
        [
            [cx - 0.5 * w, cy - 0.5 * h], # This would be the box if theta = 0
            [cx + 0.5 * w, cy - 0.5 * h],
            [cx + 0.5 * w, cy + 0.5 * h],
            [cx - 0.5 * w, cy + 0.5 * h],
            # repeat the first point to close the loop
            [cx - 0.5 * w, cy - 0.5 * h]
        ]
    )
    rotation = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    corners = np.matmul(original_points - centre, rotation) + centre
    return corners


def get_subject_image(subject, frame=7):
    '''
        Fetch the subject image from Panoptes (Zooniverse database)

        Inputs
        ------
        subject : int
            Zooniverse subject ID
        frame : int
            Frame to extract (between 0-14, default 7)

        Outputs
        -------
        img : numpy.ndarray
            RGB image corresponding to `frame`
    '''
    # get the subject metadata from Panoptes
    subjecti = Subject(int(subject))
    try:
        frame0_url = subjecti.raw['locations'][frame]['image/png']
    except KeyError:
        frame0_url = subjecti.raw['locations'][frame]['image/jpeg']

    img = io.imread(frame0_url)

    # for subjects that have an odd size, resize them
    if img.shape[0] != 1920:
        meta_width=float(subjecti.raw['metadata']['#width'])
        meta_height=float(subjecti.raw['metadata']['#height'])
        img = transform.resize(img, (meta_height, meta_width))

    return img

def sigma_shape(params, sigma):
    '''
        calculate the upper and lower bounding box
        based on the sigma of the cluster

        Inputs
        ------
        params : list
            Parameter list corresponding to the box (x, y, w, h, a).
            See `get_box_edges`
        sigma : float
            Confidence of the box, given by the minimum distance of the
            SHGO box averaging step. See the `panoptes_aggregation` module.

        Outputs
        -------
        plus_sigma : list
            Parameters corresponding to the box scaled to the upper sigma bound
        minus_sigma : list
            Parameters corresponding to the box scaled to the lower sigma bound
    '''
    gamma = np.sqrt(1 - sigma)
    plus_sigma = scale_shape(params, 1 / gamma)
    minus_sigma = scale_shape(params, gamma)
    return plus_sigma, minus_sigma

def scale_shape(params, gamma):
    '''
        scale the box by a factor of gamma
        about the center

        Inputs
        ------
        params : list
            Parameter list corresponding to the box (x, y, w, h, a).
            See `get_box_edges`
        gamma : float
            Scaling parameter. Equal to sqrt(1 - sigma), where sigma
            is the box confidence from the SHGO box-averaging step

        Outputs
        -------
        scaled_params : list
            Parameter corresponding to the box scaled by the factor gamma
    '''
    return [
        # upper left corner moves
        params[0] + (params[2] * (1 - gamma) / 2),
        params[1] + (params[3] * (1 - gamma) / 2),
        # width and height scale
        gamma * params[2],
        gamma * params[3],
        # angle does not change
        params[4]
    ]

def create_gif(jets):
    '''
        Create a gif of the jet objects showing the
        image and the plots from the `Jet.plot()` method

        Inputs
        ------
        jets : list
            List of `Jet` objects corresponding to the same subject
    '''
    # get the subject that the jet belongs to
    subject = jets[0].subject

    # create a temp plot so that we can get a size estimate
    fig, ax = plt.subplots(1, 1, dpi=150)
    ax.imshow(get_subject_image(subject, 0))
    ax.axis('off')
    fig.tight_layout()

    # loop through the frames and plot
    ims = []
    for i in range(15):
        img = get_subject_image(subject, i)

        # first, plot the image
        im1 = ax.imshow(img)

        # for each jet, plot all the details
        # and add each plot artist to the list
        jetims = []
        for jet in jets:
            jetims.extend(jet.plot(ax, plot_sigma=False))

        # combine all the plot artists together
        ims.append([im1, *jetims])

    # save the animation as a gif
    ani = animation.ArtistAnimation(fig, ims)
    ani.save(f'{subject}.gif', writer='imagemagick')
    
    
    def plot_subject_both(self, subject):
        '''
            Plots both tasks for a given subject

            Inputs
            ------
            subject : int
                Zooniverse subject ID
        '''
        fig, ax = plt.subplots(1, 1, dpi=150)

        self.plot_subject(subject, 'T1', ax)
        self.plot_subject(subject, 'T5', ax)

        fig.tight_layout()
        plt.show()

    def plot_subject(self, subject, task, ax=None):
        '''
            Plot the data for a given subject/task

            Inputs
            ------
            subject : int
                Zooniverse subject ID
            task : string
                task for the Zooniverse workflow (T1 for first jet and T2 for second jet)
            ax : matplotlib.Axes
                pass an axis variable to append the subject to a given axis (e.g., when
                making multi-subject plots where each axis corresponds to a subject).
                Default is None, and will create a new figure/axis combo.
        '''

        # get the points data and associated cluster
        points_data, points_clusters = self.get_points_data(subject, task)

        x0_i = points_data['x_start']
        y0_i = points_data['y_start']
        x1_i = points_data['x_end']
        y1_i = points_data['y_end']

        cx0_i = points_clusters['x_start']
        cy0_i = points_clusters['y_start']
        p0_i = points_clusters['prob_start']
        cx1_i = points_clusters['x_end']
        cy1_i = points_clusters['y_end']
        p1_i = points_clusters['prob_end']

        # do the same for the box
        box_data, box_clusters = self.get_box_data(subject, task)
        x_i = box_data['x']
        y_i = box_data['y']
        w_i = box_data['w']
        h_i = box_data['h']
        a_i = box_data['a']

        cx_i = box_clusters['x']
        cy_i = box_clusters['y']
        cw_i = box_clusters['w']
        ch_i = box_clusters['h']
        ca_i = box_clusters['a']
        sg_i = box_clusters['sigma']
        pb_i = box_clusters['prob']

        img = get_subject_image(subject)

        if ax is None:
            fig, ax = plt.subplots(1, 1, dpi=150)
            plot = True
        else:
            plot = False

        # plot the subject
        ax.imshow(img)

        # plot the raw classifications using a .
        alphai = np.asarray(p0_i)*0.5 + 0.5
        ax.scatter(x0_i, y0_i, 5.0, marker='.', color='blue', alpha=alphai)

        # and the end points with a yellow .
        alphai = np.asarray(p1_i)*0.5 + 0.5
        ax.scatter(x1_i, y1_i, 5.0, marker='.', color='yellow', alpha=alphai)

        # plot the clustered start/end with an x
        ax.scatter(cx0_i, cy0_i, 10.0, marker='x', color='blue')
        ax.scatter(cx1_i, cy1_i, 10.0, marker='x', color='yellow')

        # plot the raw boxes with a gray line
        for j in range(len(x_i)):
            points = get_box_edges(
                x_i[j], y_i[j], w_i[j], h_i[j], np.radians(a_i[j]))
            linewidthi = 0.2*pb_i[j]+0.1
            ax.plot(points[:, 0], points[:, 1], '-',
                    color='limegreen', linewidth=linewidthi)

        # plot the clustered box in blue
        for j in range(len(cx_i)):
            clust = get_box_edges(
                cx_i[j], cy_i[j], cw_i[j], ch_i[j], np.radians(ca_i[j]))

            # calculate the bounding box for the cluster confidence
            plus_sigma, minus_sigma = sigma_shape(
                [cx_i[j], cy_i[j], cw_i[j], ch_i[j], np.radians(ca_i[j])], sg_i[j])

            # get the boxes edges
            plus_sigma_box = get_box_edges(*plus_sigma)
            minus_sigma_box = get_box_edges(*minus_sigma)

            # create a fill between the - and + sigma boxes
            x_p = plus_sigma_box[:, 0]
            y_p = plus_sigma_box[:, 1]
            x_m = minus_sigma_box[:, 0]
            y_m = minus_sigma_box[:, 1]
            ax.fill(
                np.append(x_p, x_m[::-1]), np.append(y_p, y_m[::-1]), color='white', alpha=0.3)

            ax.plot(clust[:, 0], clust[:, 1], '-', linewidth=0.85, color='white')

        ax.axis('off')

        ax.set_xlim((0, img.shape[1]))
        ax.set_ylim((img.shape[0], 0))

        if plot:
            fig.tight_layout()
            plt.show()
