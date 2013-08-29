{:title "crsmithdev.com"}

[:div
 [:div.row
  [:div.col-md-12
   [:h1 "[untitled]"]]]
 [:div.row
  [:div.col-md-6
   [:p.lead.text-muted
    "Code and other things."]]]
 [:div.row
  [:div.col-md-12
   [:p
    "I'm Chris, a software engineer from the midwest now living and working in San Francisco. "
    "I spend my days at " [:a {:href "http://cir.ca"} "Circa"] ", where I design, develop and scale APIs and distributed systems. "
    "I'm passionate about building things and solving problems, and my current tools of choice are Python, Scala, and Clojure. "
    "Here you'll find my " [:a {:href "/blog.html"} "blog"] ", a few of the " [:a {:href "/projects.html"} "projects"] " I've worked on, "
    "and little more " [:a {:href "/about.html"} "about"] " myself."]
   [:p
    "I'm active on " [:a {:href "http://twitter.com/crsmithdev"} "Twitter"] ", " [:a {:href "http://github.com/crsmithdev"} "Github"]
    " and " [:a {:href "http://www.linkedin.com/in/crsmithdev"} "LinkedIn" ] "."]]]
 [:br]
 [:div.row
  [:div.col-md-6
   [:h4 "Recent Blog Posts"]
   (map #(let [f % url (static.core/post-url f)
               [metadata _] (static.io/read-doc f)
               date (static.core/parse-date
                     "yyyy-MM-dd" "dd MMMM yyyy"
                     (re-find #"\d*-\d*-\d*" (str f)))]
      [:div
       [:div [:a {:href url} (:title metadata)]
       [:div date]]])
      (take 8 (reverse (static.io/list-files :posts))))]
    [:div.col-md-6
    [:h4 "Recent Activity on Github"]
    [:div.gh-recent]]]]
