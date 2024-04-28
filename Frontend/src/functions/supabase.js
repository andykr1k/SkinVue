import { createClient } from "@supabase/supabase-js";
const url =
  import.meta.env.VITE_SUPABASE_URL || process.env.REACT_APP_SUPABASE_URL;
const key =
  import.meta.env.VITE_SUPABASE_KEY || process.env.REACT_APP_SUPABASE_KEY;

export const supabase = createClient(url, key);

export async function getImageUrl() {
  try {
    const { data, error } = await supabase
      .from("data")
      .select("*")
      .order("created_at", { ascending: true });

    if (error) {
      throw error;
    }
    let imageUrl = "https://whqnperemsymnmpfsoyi.supabase.co/storage/v1/object/public/pictures/" +
      data[0].picture_name;
    return imageUrl;
  } catch (error) {
    console.error("Error fetching data:", error.message);
    return null;
  }
}

export async function updateData(name, value) {
  try {
  const { data, error } = await supabase
    .from('data')
    .update({ prediction: value })
    .eq('picture_name', name)
    .select()     
    if (error) {
      throw error;
    }
    return data;
  } catch (error) {
    console.error("Error fetching data:", error.message);
    return null;
  }
}

export async function fetchData() {
  try {
    const { data, error } = await supabase
      .from("data")
      .select("*")
      .order("created_at", { ascending: true });

    if (error) {
      throw error;
    }

    return data;
  } catch (error) {
    console.error("Error fetching data:", error.message);
    return null;
  }
}

// const SubmitFeedback = async (e) => {
// e.preventDefault();
// try {
//     const { data, error } = await supabase.from("Feedback").insert([
//     {
//         first_name: firstName,
//         last_name: lastName,
//         type: selectedOption,
//         description: feedbackMessage,
//         email: email,
//     },
//     ]);

//     if (error) {
//     console.error(error);
//     } else {
//     toast.success("Thank you! We really appreciate your feedback!");
//     }
// } catch (error) {
//     console.error(error);
// }
// };
